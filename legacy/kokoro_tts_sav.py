#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kokoro-82M local neural text-to-speech (TTS) client.

Generates speech locally with the Kokoro-82M model and either streams it to
``ffplay`` or exports sections of an EPUB book or TXT file as FLAC or MP3 files.

Features:
    * Text from the command line, standard input, or TXT/EPUB files.
    * Output text directly to audio file via ``--output <filename>``.
    * Optional karaoke-style word highlighting synchronized with the audio.
    * EPUB/TXT files: read them aloud (streaming) or export audio files,
      as FLAC or as variable-bitrate MP3 (``--format``). An interrupted export
      can be resumed: finished sections/chunks are skipped.
    * Word analysis (``--analyze``): extracts unknown/out-of-dictionary
      words into a ``.dict`` file formatted as ``word|word``, utilizing
      auto-discovered Hunspell/MySpell dictionaries where available.
    * Pronunciation dictionary (``--dictionary``): a text file whose lines
      are ``word|replacement``, used to fix mispronounced words. A
      replacement between square brackets is read as phonemes, for example
      ``mr|[mˈɪstəɹ]``.
    * Runs on the CPU by default. The GPU is optional (``--device``); when it
      is busy or full, a clear message is shown instead of a crash.

Requirements:
    pip install kokoro soundfile numpy torch
    pip install ebooklib beautifulsoup4    # optional, EPUB support only
    pip install hunspell                   # optional, spelling analysis
    ``ffplay`` (part of FFmpeg) must be installed for playback, and
    ``ffmpeg`` built with libmp3lame for MP3 export.

Exit status:
    0 on success, when the user presses Ctrl+C, or when the GPU has no free
    memory (see ``EXIT_GPU_BUSY``); 1 on any other error.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import os
import queue
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import warnings
from dataclasses import dataclass

# Hide only the two known, harmless warnings emitted by PyTorch/Kokoro.
warnings.filterwarnings(
    "ignore",
    message="dropout option adds dropout",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*weight_norm",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.jit\.script",
    category=FutureWarning,
)

try:
    import numpy as np
    import soundfile as sf
    import torch
    from kokoro import KPipeline
except ImportError as exc:
    print(
        f"Error: a required package is missing ({exc}). "
        "Run 'pip install kokoro soundfile numpy torch'",
        file=sys.stderr,
    )
    sys.exit(1)

# Optional imports for EPUB processing.
try:
    import ebooklib
    from bs4 import BeautifulSoup, NavigableString, Tag
    from ebooklib import epub

    EPUB_SUPPORT = True
except ImportError:
    EPUB_SUPPORT = False

# Optional spell checking and CJK tokenization packages.
try:
    import hunspell
except ImportError:
    hunspell = None

try:
    import MeCab
except ImportError:
    MeCab = None

try:
    import jieba
except ImportError:
    jieba = None

SAMPLE_RATE = 24000  # Output sample rate of Kokoro-82M, in Hz.
REPO_ID = "hexgrad/Kokoro-82M"
TITLE_SPEED = 0.8  # Section titles are read slower than the body text.
TITLE_PAUSE_SECONDS = 0.8  # Silence between a section title and its body.
MIN_SECTION_CHARS = 10  # Shorter sections are considered empty.
MAX_CHUNK_CHARS = 400  # Target size of the text chunks used when exporting.
GPU_MIN_FREE_MIB = 1024  # Free VRAM required to use the GPU (Kokoro ~830).
AUDIO_FORMATS = ("flac", "mp3")  # Formats available for export.
DEFAULT_MP3_QUALITY = 4  # LAME VBR quality: 0 = best/largest, 9 = smallest.
DICTIONARY_SEPARATOR = "|"  # Separates the two columns of the dictionary.
ENGLISH_LANG_CODES = ("a", "b")  # Kokoro pipelines that use English G2P.
MAX_PHONEMES = 510  # Longest phoneme string that Kokoro accepts at once.
PHONEME_CHUNK_CHARS = 300  # Text chunk size when phonemes are injected.

# Inline phoneme syntax of Kokoro: [word](/phonemes/)
PHONEME_MARKUP_RE = re.compile(r"\[(\w+)\]\(/([^/()\[\]\n]*)/\)")

# HTML elements that start a new block of text or a separate token.
TEXT_BREAK_TAGS = frozenset(
    {
        "address", "article", "aside", "blockquote", "body", "br",
        "caption", "dd", "details", "div", "dl", "dt", "fieldset",
        "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4",
        "h5", "h6", "head", "header", "hr", "html", "li", "main", "nav",
        "ol", "p", "pre", "section", "sub", "summary", "sup", "table",
        "tbody", "td", "tfoot", "th", "thead", "title", "tr", "ul",
    }
)
EXIT_GPU_BUSY = 0  # Exit status when the GPU has no free memory.

# Mapping ISO language codes to Kokoro internal pipeline codes.
SUPPORTED_LANGUAGES = {
    "es": {"internal": "e", "name": "Spanish (Español)", "hunspell_locales": ["es_ES", "es_AR", "es_MX", "es"]},
    "en": {"internal": "b", "name": "English (Defaults to British)", "hunspell_locales": ["en_GB", "en_US", "en"]},
    "uk": {"internal": "b", "name": "English - British (UK)", "hunspell_locales": ["en_GB", "en"]},
    "us": {"internal": "a", "name": "English - American (US)", "hunspell_locales": ["en_US", "en"]},
    "fr": {"internal": "f", "name": "French (Français)", "hunspell_locales": ["fr_FR", "fr"]},
    "it": {"internal": "i", "name": "Italian (Italiano)", "hunspell_locales": ["it_IT", "it"]},
    "ja": {"internal": "j", "name": "Japanese (日本語)", "hunspell_locales": []},
    "zh": {"internal": "z", "name": "Mandarin Chinese (中文)", "hunspell_locales": []},
    "pt": {"internal": "p", "name": "Portuguese (Português)", "hunspell_locales": ["pt_PT", "pt_BR", "pt"]},
}

# Official Kokoro-82M voices grouped by language category.
ALL_VOICES = {
    "es (Spanish)": [
        "ef_dora",
        "em_alex",
        "em_santa",
    ],
    "en / uk (British English)": [
        "bf_emma",
        "bf_isabella",
        "bf_alice",
        "bf_lily",
        "bm_george",
        "bm_lewis",
        "bm_daniel",
        "bm_fable",
    ],
    "us (American English)": [
        "af_heart",
        "af_alloy",
        "af_aoede",
        "af_bella",
        "af_jessica",
        "af_kore",
        "af_nicole",
        "af_nova",
        "af_river",
        "af_sarah",
        "af_sky",
        "am_adam",
        "am_echo",
        "am_eric",
        "am_fenrir",
        "am_liam",
        "am_michael",
        "am_onyx",
        "am_puck",
        "am_santa",
    ],
    "fr (French)": [
        "ff_siwis",
    ],
    "it (Italian)": [
        "if_sara",
        "im_nicola",
    ],
    "ja (Japanese)": [
        "jf_alpha",
        "jf_gongitsune",
        "jf_nezumi",
        "jf_tebukuro",
        "jm_kumo",
    ],
    "zh (Mandarin Chinese)": [
        "zf_xiaobei",
        "zf_xiaoni",
        "zf_xiaoxiao",
        "zf_xiaoyi",
        "zm_yunjian",
        "zm_yunxi",
        "zm_yunxia",
        "zm_yunyang",
    ],
    "pt (Portuguese)": [
        "pf_dora",
        "pm_alex",
        "pm_santa",
    ],
}

ISO_TO_VOICE_KEY = {
    "es": "es (Spanish)",
    "en": "en / uk (British English)",
    "uk": "en / uk (British English)",
    "us": "us (American English)",
    "fr": "fr (French)",
    "it": "it (Italian)",
    "ja": "ja (Japanese)",
    "zh": "ja (Japanese)",
    "pt": "pt (Portuguese)",
}

KNOWN_VOICES = frozenset(
    voice for voices in ALL_VOICES.values() for voice in voices
)

_ENGLISH_TEST_TEXT = (
    "Hello, I am an AI-synthesized voice, pleased to greet you. "
    "I hope you make good use of me from now on."
)

TEST_TEXTS = {
    "es": (
        "Hola, soy una voz sintetizada por IA, encantado de saludarte. "
        "Espero que a partir de ahora hagas un buen uso de mí."
    ),
    "en": _ENGLISH_TEST_TEXT,
    "uk": _ENGLISH_TEST_TEXT,
    "us": _ENGLISH_TEST_TEXT,
    "fr": (
        "Bonjour, je suis une voix synthétisée par IA, enchanté de vous "
        "saluer. J'espère que vous ferez bon usage de moi à l'avenir."
    ),
    "it": (
        "Ciao, sono una voce sintetizzata dall'IA, felice di salutarti. "
        "Spero che d'ora in poi farai buon uso di me."
    ),
    "ja": (
        "こんにちは、私はAIによって合成された音声です。お会いできて嬉しいです。"
        "これから上手に活用していただければ幸いです。"
    ),
    "zh": (
        "你好，我是由人工智能合成的声音，很高兴见到你。"
        "希望从现在开始你能好好利用我。"
    ),
    "pt": (
        "Olá, sou uma voz sintetizada por IA, prazer em conhec-lo. "
        "Espero que faça bom uso de mim a partir de agora."
    ),
}


class TTSError(Exception):
    """Error raised while synthesizing or playing audio."""


class PlayerNotFoundError(TTSError):
    """Raised when ``ffplay`` is not installed (fatal for playback)."""


class GpuMemoryError(TTSError):
    """Raised when the GPU has no free memory for the model."""


@dataclass
class Section:
    index: int
    title: str
    has_heading: bool
    soup: BeautifulSoup | None = None
    body_text: str | None = None


# --------------------------------------------------------------------------
# Auto-discovery for Hunspell / MySpell and Language Processing
# --------------------------------------------------------------------------

def find_hunspell_dictionary(lang_code: str) -> str | None:
    """Automatically discover Hunspell/MySpell dictionary (.dic/.aff) path prefix."""
    lang_info = SUPPORTED_LANGUAGES.get(lang_code.lower())
    if not lang_info:
        return None

    locales = lang_info.get("hunspell_locales", [])
    search_dirs = [
        "/usr/share/hunspell",
        "/usr/share/myspell",
        "/usr/local/share/hunspell",
        "/usr/local/share/myspell",
        os.path.expanduser("~/.local/share/hunspell"),
        os.path.expanduser("~/.local/share/myspell"),
        os.path.expanduser("~/.hunspell"),
        os.getcwd()
    ]

    for locale in locales:
        for d in search_dirs:
            if not os.path.exists(d):
                continue
            aff_path = os.path.join(d, f"{locale}.aff")
            dic_path = os.path.join(d, f"{locale}.dic")
            if os.path.exists(aff_path) and os.path.exists(dic_path):
                return os.path.join(d, locale)
    return None


class LanguageProcessor:
    """Handles tokenization, word extraction, and spelling checks via Hunspell/MySpell or CJK stubs."""

    def __init__(self, lang_code: str) -> None:
        self.lang_code = lang_code.lower()
        self.checker = None
        self.tagger = None
        self._init_backend()

    def _init_backend(self) -> None:
        if self.lang_code in ("ja", "japanese"):
            if MeCab is not None:
                with contextlib.suppress(Exception):
                    self.tagger = MeCab.Tagger("-Ochasen")
            return

        if self.lang_code in ("zh", "chinese"):
            return

        if hunspell is None:
            print(
                "[WARNING] Python 'hunspell' package is not installed. "
                "Install it with 'pip install hunspell' for spell checking.",
                file=sys.stderr,
            )
            return

        prefix = find_hunspell_dictionary(self.lang_code)
        if prefix:
            aff_path = f"{prefix}.aff"
            dic_path = f"{prefix}.dic"
            try:
                self.checker = hunspell.HunSpell(dic_path, aff_path)
                print(f"[INFO] Auto-loaded dictionary from: {prefix}")
            except Exception as exc:
                print(f"[WARNING] Could not initialize Hunspell with {prefix}: {exc}", file=sys.stderr)
        else:
            print(
                f"[WARNING] No Hunspell/MySpell dictionary found for language '{self.lang_code}'.",
                file=sys.stderr,
            )

    def extract_words(self, text: str) -> list[str]:
        if self.lang_code in ("ja", "japanese"):
            if self.tagger is not None:
                node = self.tagger.parseToNode(text)
                words = []
                while node:
                    surface = node.surface.strip()
                    if surface:
                        words.append(surface)
                    node = node.next
                return words
            return [c for c in text if not c.isspace()]

        if self.lang_code in ("zh", "chinese"):
            if jieba is not None:
                return list(jieba.cut(text))
            return [c for c in text if not c.isspace()]

        return re.findall(r"\b\w+\b", text)

    def check_word(self, word: str) -> bool:
        if self.lang_code in ("ja", "japanese", "zh", "chinese"):
            return True
        if self.checker is not None:
            return self.checker.spell(word)
        return True


def get_default_voice(lang_code: str) -> str:
    voice_key = ISO_TO_VOICE_KEY.get(lang_code.lower())
    if voice_key and ALL_VOICES.get(voice_key):
        return ALL_VOICES[voice_key][0]
    return "ef_dora"


def validate_voice(lang_code: str, voice: str | None) -> None:
    if not voice or os.path.isfile(voice):
        return
    if voice not in KNOWN_VOICES:
        raise TTSError(
            f"Unknown voice '{voice}'. "
            "Use --list-voices to view the available voices."
        )
    internal = SUPPORTED_LANGUAGES[lang_code.lower()]["internal"]
    if voice[0] != internal:
        print(
            f"Warning: voice '{voice}' belongs to a different language "
            f"than --lang {lang_code}.",
            file=sys.stderr,
        )


def show_languages() -> None:
    print("\nSupported ISO Languages:")
    print("-" * 50)
    for code, info in SUPPORTED_LANGUAGES.items():
        print(f"  [{code}] -> {info['name']}")
    print()


def show_voices(lang_filter: str = "all") -> None:
    code = lang_filter.lower()
    if code == "all":
        print("\nVerified Official List of Neural Voices:")
        print("-" * 45)
        for category, voices in ALL_VOICES.items():
            print(f"  {category}:")
            for voice in voices:
                print(f"    - {voice}")
        print()
        return

    voice_key = ISO_TO_VOICE_KEY.get(code)
    if not voice_key or voice_key not in ALL_VOICES:
        raise TTSError(
            f"Unsupported language code '{lang_filter}' for voices. "
            "Use --list-languages to view available options."
        )

    print(f"\nNeural Voices for language code '[{code}]':")
    print("-" * 45)
    print(f"  {voice_key}:")
    for voice in ALL_VOICES[voice_key]:
        print(f"    - {voice}")
    print()


# --------------------------------------------------------------------------
# Pronunciation dictionary and IPA support
# --------------------------------------------------------------------------


def parse_phonemes(spoken: str) -> str | None:
    """Return the phonemes of a replacement written as ``[phonemes]``."""
    text = spoken.strip()
    if len(text) < 2 or text[0] != "[" or text[-1] != "]":
        return None
    phonemes = text[1:-1].strip()
    if re.search(r"[\[\]()/\n]", phonemes):
        raise ValueError("phonemes cannot contain any of [ ] ( ) /")
    return phonemes


class PronunciationDictionary:
    """Replace words in text so they are pronounced as desired."""

    def __init__(self, entries: dict, ignore_case: bool = False) -> None:
        entries = {
            word: self._to_spoken(word, spoken)
            for word, spoken in entries.items()
            if word
        }
        self._ignore_case = ignore_case
        self._replacements = {
            self._key(word): spoken for word, spoken in entries.items()
        }
        alternatives = [
            self._word_regex(word)
            for word in sorted(entries, key=len, reverse=True)
        ]
        flags = re.IGNORECASE if ignore_case else 0
        self._pattern = (
            re.compile("|".join(alternatives), flags)
            if alternatives
            else None
        )

    def __len__(self) -> int:
        return len(self._replacements)

    def apply(self, text: str) -> str:
        if self._pattern is None:
            return text
        return self._pattern.sub(self._replace, text)

    @staticmethod
    def _to_spoken(word: str, spoken: str) -> str:
        phonemes = parse_phonemes(spoken)
        if phonemes is None:
            return spoken
        if not phonemes:
            return ""
        label = re.sub(r"[\W_]+", "", word) or "x"
        return f"[{label}](/{phonemes}/)"

    def _key(self, word: str) -> str:
        return word.casefold() if self._ignore_case else word

    def _replace(self, match: re.Match) -> str:
        word = match.group(0)
        return self._replacements.get(self._key(word), word)

    @staticmethod
    def _word_regex(word: str) -> str:
        start = r"(?<!\w)" if re.match(r"\w", word[0]) else ""
        end = r"(?!\w)" if re.match(r"\w", word[-1]) else ""
        return f"{start}{re.escape(word)}{end}"


def load_dictionary(
    path: str, ignore_case: bool = False
) -> PronunciationDictionary:
    entries: dict = {}
    try:
        with open(path, encoding="utf-8-sig") as handle:
            for number, raw_line in enumerate(handle, 1):
                line = raw_line.rstrip("\r\n")
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                columns = line.split(DICTIONARY_SEPARATOR)
                word = columns[0].strip()
                if len(columns) < 2 or not word:
                    print(
                        f"Warning: {path}:{number}: expected "
                        f"'word{DICTIONARY_SEPARATOR}replacement'; "
                        "line skipped.",
                        file=sys.stderr,
                    )
                    continue
                spoken = columns[1].strip()
                try:
                    parse_phonemes(spoken)
                except ValueError as exc:
                    print(
                        f"Warning: {path}:{number}: {exc}; line skipped.",
                        file=sys.stderr,
                    )
                    continue
                entries[word] = spoken
    except OSError as exc:
        raise TTSError(f"Cannot read dictionary '{path}': {exc}") from exc
    except UnicodeDecodeError as exc:
        raise TTSError(
            f"Dictionary '{path}' is not valid UTF-8: {exc}"
        ) from exc
    return PronunciationDictionary(entries, ignore_case=ignore_case)


# --------------------------------------------------------------------------
# Device selection & Audio helpers
# --------------------------------------------------------------------------

def is_cuda_oom(exc: BaseException) -> bool:
    oom_class = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_class is not None and isinstance(exc, oom_class):
        return True
    return isinstance(exc, RuntimeError) and (
        "out of memory" in str(exc).lower()
    )


def gpu_oom_message(
    free_mib: int | None = None, total_mib: int | None = None
) -> str:
    detail = ""
    if free_mib is not None and total_mib is not None:
        detail = (
            f" Only {free_mib} MiB are free out of {total_mib} MiB "
            f"(at least {GPU_MIN_FREE_MIB} MiB are needed)."
        )
    return (
        "The GPU does not have enough free memory to generate audio."
        f"{detail} Close other programs that use the GPU (see nvidia-smi) "
        "and try again, or use --device cpu."
    )


def gpu_has_room() -> tuple[bool, str]:
    try:
        free, total = torch.cuda.mem_get_info()
    except Exception as exc:
        if is_cuda_oom(exc):
            return False, gpu_oom_message()
        return True, ""
    free_mib, total_mib = free // 2**20, total // 2**20
    if free_mib < GPU_MIN_FREE_MIB:
        return False, gpu_oom_message(free_mib, total_mib)
    return True, ""


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        if requested == "cuda":
            raise TTSError(
                "CUDA was requested but no usable GPU was found. "
                "Use --device cpu or --device auto."
            )
        return "cpu"

    has_room, message = gpu_has_room()
    if has_room:
        return "cuda"
    if requested == "auto":
        print(f"[INFO] {message} Falling back to the CPU.", file=sys.stderr)
        return "cpu"
    raise GpuMemoryError(message)


@functools.lru_cache(maxsize=None)
def get_pipeline(internal_lang_code: str, device: str) -> KPipeline:
    try:
        return KPipeline(
            lang_code=internal_lang_code,
            repo_id=REPO_ID,
            device=device,
        )
    except Exception as exc:
        if is_cuda_oom(exc):
            raise GpuMemoryError(gpu_oom_message()) from exc
        raise


def to_float32(audio) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32)


def to_pcm16(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def streaming_wav_header(
    sample_rate: int = SAMPLE_RATE, channels: int = 1, bits: int = 16
) -> bytes:
    block_align = channels * bits // 8
    byte_rate = sample_rate * block_align
    unknown_size = 0xFFFFFFFF
    return (
        b"RIFF"
        + struct.pack("<I", unknown_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack(
            "<IHHIIHH", 16, 1, channels, sample_rate, byte_rate,
            block_align, bits,
        )
        + b"data"
        + struct.pack("<I", unknown_size)
    )


def split_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list:
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+|(?<=[。！？])\s*", text)
        if s.strip()
    ]
    chunks: list = []
    current = ""
    for sentence in sentences:
        if current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def require_ffplay() -> None:
    if shutil.which("ffplay") is None:
        raise PlayerNotFoundError(
            "'ffplay' (from ffmpeg) is not installed."
        )


class AudioPlayer:
    COMMAND = [
        "ffplay",
        "-nodisp",
        "-autoexit",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
    ]

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> AudioPlayer:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        proc = self._proc
        if proc is None:
            return
        self._close_stdin()
        if exc_type is None:
            proc.wait()
            return
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    def write(self, samples: np.ndarray) -> None:
        payload = to_pcm16(samples)
        if self._proc is None:
            self._start()
            payload = streaming_wav_header() + payload
        try:
            self._proc.stdin.write(payload)
            self._proc.stdin.flush()
        except OSError as exc:
            raise TTSError(
                "ffplay stopped unexpectedly "
                f"(exit code {self._proc.poll()})."
            ) from exc

    def _start(self) -> None:
        try:
            self._proc = subprocess.Popen(
                self.COMMAND, stdin=subprocess.PIPE
            )
        except FileNotFoundError as exc:
            raise PlayerNotFoundError(
                "'ffplay' (from ffmpeg) is not installed."
            ) from exc

    def _close_stdin(self) -> None:
        stdin = self._proc.stdin
        if stdin is not None:
            with contextlib.suppress(OSError):
                stdin.close()


def require_mp3_encoder() -> None:
    if shutil.which("ffmpeg") is None:
        raise TTSError(
            "'ffmpeg' is required for MP3 output but is not installed."
        )
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    if "libmp3lame" not in result.stdout:
        raise TTSError("This ffmpeg build has no MP3 encoder (libmp3lame).")


class Mp3Writer:
    def __init__(self, path: str, quality: int = DEFAULT_MP3_QUALITY) -> None:
        self._path = path
        self._quality = quality
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> Mp3Writer:
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
            "-i", "pipe:0",
            "-c:a", "libmp3lame", "-q:a", str(self._quality),
            "-f", "mp3", self._path,
        ]
        try:
            self._proc = subprocess.Popen(
                command, stdin=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except FileNotFoundError as exc:
            raise TTSError("'ffmpeg' is not installed.") from exc
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        proc = self._proc
        if proc is None:
            return
        if exc_type is not None:
            self._stop()
            return
        self._proc = None
        _, error = proc.communicate()
        if proc.returncode != 0:
            raise TTSError(
                "ffmpeg failed to encode the MP3 file: "
                f"{error.decode(errors='replace').strip()}"
            )

    def write(self, samples: np.ndarray) -> None:
        try:
            self._proc.stdin.write(to_pcm16(samples))
        except OSError as exc:
            raise TTSError(
                "ffmpeg stopped unexpectedly while encoding the MP3 "
                f"file: {self._stop()}"
            ) from exc

    def _stop(self) -> str:
        proc, self._proc = self._proc, None
        if proc is None:
            return ""
        proc.kill()
        _, error = proc.communicate()
        return error.decode(errors="replace").strip()


def open_audio_writer(path: str, audio_format: str, mp3_quality: int):
    if audio_format == "mp3":
        return Mp3Writer(path, mp3_quality)
    return sf.SoundFile(
        path,
        mode="w",
        samplerate=SAMPLE_RATE,
        channels=1,
        format="FLAC",
        subtype="PCM_16",
    )


def wrap_words(words: list, width: int) -> list:
    lines: list = [[]]
    length = 0
    for index, word in enumerate(words):
        extra = len(word) + (1 if lines[-1] else 0)
        if lines[-1] and length + extra > width:
            lines.append([])
            length = 0
            extra = len(word)
        lines[-1].append(index)
        length += extra
    return lines


class KaraokeDisplay:
    PREFIX = "-> "
    HIGHLIGHT = "\033[1;36m"
    RESET = "\033[0m"

    def __init__(self) -> None:
        self._interactive = sys.stdout.isatty()
        self._queue: queue.Queue = queue.Queue()
        self._abort = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, text: str, duration: float) -> None:
        self._queue.put((text, duration))

    def close(self, abort: bool = False) -> None:
        if abort:
            self._abort.set()
        self._queue.put(None)
        self._thread.join()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            if not self._abort.is_set():
                self._show(*item)

    def _show(self, text: str, duration: float) -> None:
        words = text.split()
        if not self._interactive or not words:
            print(f"{self.PREFIX}{text}", flush=True)
            return

        columns = shutil.get_terminal_size().columns
        lines = wrap_words(words, max(20, columns - len(self.PREFIX) - 1))
        total_chars = sum(len(word) for word in words)
        for current, word in enumerate(words):
            if self._abort.is_set():
                break
            self._draw(words, lines, current, first=current == 0)
            time.sleep(max(0.05, duration * len(word) / total_chars))
        self._draw(words, lines, None, first=False)

    def _draw(
        self, words: list, lines: list, current: int | None, first: bool
    ) -> None:
        if not first:
            sys.stdout.write(f"\033[{len(lines)}F")
        rows = []
        for number, line in enumerate(lines):
            prefix = self.PREFIX if number == 0 else " " * len(self.PREFIX)
            parts = []
            for index in line:
                if index == current:
                    parts.append(
                        f"{self.HIGHLIGHT}{words[index]}{self.RESET}"
                    )
                else:
                    parts.append(words[index])
            rows.append(prefix + " ".join(parts) + "\033[K")
        sys.stdout.write("\n".join(rows) + "\n")
        sys.stdout.flush()


# --------------------------------------------------------------------------
# Synthesis and Phoneme handling
# --------------------------------------------------------------------------

_WARNED: set = set()


def warn_once(message: str) -> None:
    if message not in _WARNED:
        _WARNED.add(message)
        print(f"Warning: {message}", file=sys.stderr)


def strip_phoneme_markup(text: str) -> str:
    return PHONEME_MARKUP_RE.sub(r"\1", text)


def _text_phonemes(pipeline: KPipeline, text: str) -> str:
    if not text.strip():
        return ""
    result = pipeline.g2p(text)
    phonemes = result[0] if isinstance(result, tuple) else result
    return (phonemes or "").strip()


def _join_phonemes(parts: list) -> str:
    joined = ""
    for part in parts:
        if not part:
            continue
        if joined and part[0] not in ",.;:!?…":
            joined += " "
        joined += part
    return joined


def _chunk_phonemes(pipeline: KPipeline, chunk: str) -> str | None:
    try:
        parts = []
        position = 0
        for match in PHONEME_MARKUP_RE.finditer(chunk):
            before = chunk[position:match.start()]
            parts.append(_text_phonemes(pipeline, before))
            parts.append(match.group(2).strip())
            position = match.end()
        parts.append(_text_phonemes(pipeline, chunk[position:]))
    except Exception as exc:
        warn_once(
            f"could not phonemize text around IPA entries ({exc}); "
            "they are read as plain words."
        )
        return None

    phonemes = _join_phonemes(parts)
    if len(phonemes) > MAX_PHONEMES:
        warn_once(
            "a text chunk with IPA entries is too long; its IPA entries "
            "are read as plain words."
        )
        return None
    return phonemes


def _generate_with_phonemes(
    pipeline: KPipeline, text: str, voice: str, speed: float
):
    can_inject = hasattr(pipeline, "generate_from_tokens")
    if not can_inject:
        warn_once(
            "this version of kokoro cannot synthesize phonemes directly; "
            "IPA entries are read as plain words."
        )
    for line in re.split(r"\n+", text):
        if not line.strip():
            continue
        for chunk in split_into_chunks(line, PHONEME_CHUNK_CHARS):
            phonemes = None
            if can_inject and PHONEME_MARKUP_RE.search(chunk):
                phonemes = _chunk_phonemes(pipeline, chunk)
            plain = strip_phoneme_markup(chunk)
            if phonemes is None:
                for graphemes, _, audio in pipeline(
                    plain, voice=voice, speed=speed, split_pattern=r"\n+"
                ):
                    if audio is not None:
                        yield graphemes, audio
                continue
            for _, _, audio in pipeline.generate_from_tokens(
                phonemes, voice=voice, speed=speed
            ):
                if audio is not None:
                    yield plain, audio


def generate_speech(
    pipeline: KPipeline,
    internal_lang: str,
    text: str,
    voice: str,
    speed: float,
):
    if internal_lang not in ENGLISH_LANG_CODES and (
        PHONEME_MARKUP_RE.search(text)
    ):
        yield from _generate_with_phonemes(pipeline, text, voice, speed)
        return
    for graphemes, _, audio in pipeline(
        text, voice=voice, speed=speed, split_pattern=r"\n+"
    ):
        if audio is not None:
            yield graphemes, audio


def synthesize_and_play(
    text: str,
    lang_code: str = "es",
    voice: str | None = None,
    display: bool = False,
    speed: float = 1.0,
    device: str = "cpu",
) -> None:
    lang_info = SUPPORTED_LANGUAGES.get(lang_code.lower())
    if not lang_info:
        raise TTSError(
            f"Unsupported language code '{lang_code}'. "
            "Use --list-languages to view available options."
        )

    selected_voice = voice or get_default_voice(lang_code)
    pipeline = get_pipeline(lang_info["internal"], device)
    generator = generate_speech(
        pipeline, lang_info["internal"], text, selected_voice, speed
    )
    karaoke = KaraokeDisplay() if display else None
    completed = False
    try:
        with AudioPlayer() as player:
            for graphemes, audio in generator:
                samples = to_float32(audio)
                player.write(samples)
                if karaoke is not None and graphemes.strip():
                    karaoke.submit(
                        graphemes.strip(), len(samples) / SAMPLE_RATE
                    )
        completed = True
    except TTSError:
        raise
    except Exception as exc:
        if is_cuda_oom(exc):
            torch.cuda.empty_cache()
            raise GpuMemoryError(gpu_oom_message()) from exc
        raise TTSError(
            f"Error during audio generation or playback: {exc}"
        ) from exc
    finally:
        if karaoke is not None:
            karaoke.close(abort=not completed)


def synthesize_text_to_file(
    text: str,
    output_filepath: str,
    lang_code: str = "es",
    voice: str | None = None,
    speed: float = 1.0,
    device: str = "cpu",
    audio_format: str = "flac",
    mp3_quality: int = DEFAULT_MP3_QUALITY,
) -> None:
    lang_info = SUPPORTED_LANGUAGES.get(lang_code.lower())
    if not lang_info:
        raise TTSError(f"Unsupported language code '{lang_code}'.")

    selected_voice = voice or get_default_voice(lang_code)
    pipeline = get_pipeline(lang_info["internal"], device)
    temp_filepath = output_filepath + ".tmp"
    finished = False

    try:
        samples_written = 0
        chunks = split_into_chunks(text)
        with open_audio_writer(
            temp_filepath, audio_format, mp3_quality
        ) as out:
            for number, chunk in enumerate(chunks, 1):
                for _, audio in generate_speech(
                    pipeline,
                    lang_info["internal"],
                    chunk,
                    selected_voice,
                    speed,
                ):
                    if audio is not None:
                        samples = to_float32(audio)
                        out.write(samples)
                        samples_written += len(samples)

                percent = int(number / len(chunks) * 100)
                sys.stdout.write(
                    f"\r    -> Progress: [{percent}%] "
                    f"({number}/{len(chunks)} chunks)"
                )
                sys.stdout.flush()
            print()

        if samples_written == 0:
            raise TTSError(f"No audio was generated for '{output_filepath}'.")
        os.replace(temp_filepath, output_filepath)
        finished = True
        print(f"[INFO] Audio successfully saved to '{output_filepath}'.")
    except TTSError:
        raise
    except Exception as exc:
        if is_cuda_oom(exc):
            torch.cuda.empty_cache()
            raise GpuMemoryError(gpu_oom_message()) from exc
        raise TTSError(
            f"Error generating audio for file '{output_filepath}': {exc}"
        ) from exc
    finally:
        if not finished:
            print()
            with contextlib.suppress(OSError):
                os.remove(temp_filepath)


def synthesize_epub_section_to_file(
    title_text: str,
    body_text: str,
    output_filepath: str,
    lang_code: str = "es",
    voice: str | None = None,
    speed: float = 1.0,
    device: str = "cpu",
    audio_format: str = "flac",
    mp3_quality: int = DEFAULT_MP3_QUALITY,
) -> None:
    lang_info = SUPPORTED_LANGUAGES.get(lang_code.lower())
    if not lang_info:
        raise TTSError(f"Unsupported language code '{lang_code}'.")

    selected_voice = voice or get_default_voice(lang_code)
    pipeline = get_pipeline(lang_info["internal"], device)
    temp_filepath = output_filepath + ".tmp"
    finished = False

    try:
        samples_written = 0
        with open_audio_writer(
            temp_filepath, audio_format, mp3_quality
        ) as out:
            if title_text:
                for _, audio in generate_speech(
                    pipeline,
                    lang_info["internal"],
                    title_text,
                    selected_voice,
                    TITLE_SPEED,
                ):
                    if audio is not None:
                        samples = to_float32(audio)
                        out.write(samples)
                        samples_written += len(samples)
                out.write(
                    np.zeros(
                        int(SAMPLE_RATE * TITLE_PAUSE_SECONDS),
                        dtype=np.float32,
                    )
                )

            chunks = split_into_chunks(body_text)
            for number, chunk in enumerate(chunks, 1):
                for _, audio in generate_speech(
                    pipeline,
                    lang_info["internal"],
                    chunk,
                    selected_voice,
                    speed,
                ):
                    if audio is not None:
                        samples = to_float32(audio)
                        out.write(samples)
                        samples_written += len(samples)

                percent = int(number / len(chunks) * 100)
                sys.stdout.write(
                    f"\r    -> Progress: [{percent}%] "
                    f"({number}/{len(chunks)} body chunks)"
                )
                sys.stdout.flush()
            print()

        if samples_written == 0:
            raise TTSError(f"No audio was generated for '{output_filepath}'.")
        os.replace(temp_filepath, output_filepath)
        finished = True
    except TTSError:
        raise
    except Exception as exc:
        if is_cuda_oom(exc):
            torch.cuda.empty_cache()
            raise GpuMemoryError(gpu_oom_message()) from exc
        raise TTSError(
            f"Error generating audio for file '{output_filepath}': {exc}"
        ) from exc
    finally:
        if not finished:
            print()
            with contextlib.suppress(OSError):
                os.remove(temp_filepath)


# --------------------------------------------------------------------------
# EPUB and TXT handling & Word Analysis
# --------------------------------------------------------------------------

def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r'[\\/*?:"<>|]', "", name)
    cleaned = re.sub(r"\s+", "_", cleaned).strip("_")
    return cleaned[:50] if cleaned else "section"


def _collect_text(node: Tag, parts: list) -> None:
    for child in node.children:
        if isinstance(child, Tag):
            local_name = (child.name or "").rsplit(":", 1)[-1].lower()
            separate = local_name in TEXT_BREAK_TAGS
            if separate:
                parts.append(" ")
            _collect_text(child, parts)
            if separate:
                parts.append(" ")
        elif type(child) is NavigableString:
            parts.append(str(child))


def get_plain_text(node: Tag) -> str:
    parts: list = []
    _collect_text(node, parts)
    return " ".join("".join(parts).split())


def extract_sections(epub_path: str) -> list:
    book = epub.read_epub(epub_path)
    documents = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    sections = []
    for index, item in enumerate(documents, 1):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        heading_tag = soup.find(["h1", "h2", "h3"])
        title = get_plain_text(heading_tag) if heading_tag else ""
        sections.append(
            Section(
                index=index,
                title=title or f"Section {index}",
                has_heading=bool(title),
                soup=soup,
            )
        )
    return sections


def get_section_body(section: Section) -> str:
    if section.body_text is not None:
        return section.body_text
    full_text = get_plain_text(section.soup)
    if len(full_text) < MIN_SECTION_CHARS:
        return ""
    if section.has_heading and full_text.startswith(section.title):
        return full_text[len(section.title):].strip()
    return full_text


def extract_txt_sections(txt_path: str) -> list:
    try:
        with open(txt_path, encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(txt_path, encoding="latin-1") as f:
            content = f.read()

    raw_blocks = re.split(r'\n\s*\n\s*\n+', content)
    if len(raw_blocks) <= 1:
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [content.strip()]

        sections = []
        current_chunk = []
        current_len = 0
        sec_idx = 1
        for p in paragraphs:
            current_chunk.append(p)
            current_len += len(p)
            if current_len > 2500:
                body = "\n\n".join(current_chunk)
                sections.append(Section(index=sec_idx, title=f"Part {sec_idx}", has_heading=False, body_text=body))
                sec_idx += 1
                current_chunk = []
                current_len = 0
        if current_chunk:
            body = "\n\n".join(current_chunk)
            sections.append(Section(index=sec_idx, title=f"Part {sec_idx}", has_heading=False, body_text=body))
        return sections

    sections = []
    for index, block in enumerate(raw_blocks, 1):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        has_heading = False
        title = f"Block {index}"
        body_text = block.strip()
        if len(lines[0]) < 80 and (re.match(r'^(chapter|part|section|\d+)', lines[0], re.IGNORECASE) or len(lines) > 1):
            title = lines[0]
            has_heading = True
            body_text = "\n".join(lines[1:]).strip()

        if not body_text:
            body_text = block.strip()

        sections.append(
            Section(
                index=index,
                title=title,
                has_heading=has_heading,
                body_text=body_text,
            )
        )
    return sections


def find_start_index(sections: list, section_start: str | None) -> int:
    if not section_start:
        return 1

    if section_start.isdigit():
        wanted = int(section_start)
        if any(sec.index == wanted for sec in sections):
            return wanted
    else:
        for sec in sections:
            if section_start.lower() in sec.title.lower():
                return sec.index

    raise TTSError(
        f"Section/Block '{section_start}' not found. "
        "Use --section-list to see available sections."
    )


def analyze_book_words(
    file_path: str,
    lang_code: str,
    dict_output_path: str,
    is_epub: bool,
    device: str = "cpu",
) -> int:
    """Analyze EPUB or TXT text using auto-discovered Hunspell dictionaries."""
    if os.path.exists(dict_output_path):
        raise TTSError(
            f"Target file '{dict_output_path}' already exists. "
            "Operation aborted."
        )

    lang_info = SUPPORTED_LANGUAGES.get(lang_code.lower())
    if not lang_info:
        raise TTSError(f"Unsupported language code '{lang_code}'.")

    processor = LanguageProcessor(lang_code)

    if processor.checker is not None:
        print(f"[INFO] Using strict Hunspell dictionary backend for '{lang_code}'.")
    else:
        print(
            f"[WARNING] No valid Hunspell dictionary loaded for '{lang_code}'. "
            "Falling back to fallback pipeline inspection.",
            file=sys.stderr,
        )

    file_type = "EPUB" if is_epub else "TXT"
    print(f"[INFO] Analyzing {file_type} file '{file_path}'...")

    sections = extract_sections(file_path) if is_epub else extract_txt_sections(file_path)

    all_text = []
    for sec in sections:
        all_text.append(sec.title)
        all_text.append(get_section_body(sec))

    combined_text = " ".join(all_text)
    raw_words = processor.extract_words(combined_text)

    unique_words = sorted(
        list(set(w for w in raw_words if not w.isdigit() and len(w) > 1))
    )

    unknown_words = []
    print(f"[INFO] Inspecting {len(unique_words)} distinct words...")

    if processor.checker is not None:
        for word in unique_words:
            if not processor.check_word(word):
                unknown_words.append(word)
    else:
        pipeline = get_pipeline(lang_info["internal"], device)
        for word in unique_words:
            try:
                if hasattr(pipeline, "phonemize"):
                    phonemes = pipeline.phonemize(word)
                elif hasattr(pipeline, "g2p"):
                    phonemes, _ = pipeline.g2p(word)
                else:
                    phonemes = ""
                    for _, audio in generate_speech(pipeline, lang_info["internal"], word, get_default_voice(lang_code), 1.0):
                        if audio is not None:
                            phonemes = "ok"
                            break
                if not phonemes or "?" in str(phonemes) or "UNK" in str(phonemes):
                    unknown_words.append(word)
            except Exception:
                unknown_words.append(word)

    try:
        with open(dict_output_path, "w", encoding="utf-8") as f:
            for w in unknown_words:
                f.write(f"{w}{DICTIONARY_SEPARATOR}{w}\n")
        print(
            f"[INFO] Analysis completed. Found {len(unknown_words)} "
            f"unrecognized words. File written to '{dict_output_path}'."
        )
    except OSError as exc:
        raise TTSError(
            f"Error writing to dictionary file '{dict_output_path}': {exc}"
        ) from exc
    return 0


def stream_sections(
    sections: list,
    total: int,
    lang_code: str,
    voice: str | None,
    display: bool,
    speed: float,
    device: str,
    lexicon: PronunciationDictionary,
) -> int:
    failed = 0
    for sec in sections:
        print(f"[{sec.index}/{total}] Reading section: {sec.title}")
        try:
            if sec.has_heading:
                synthesize_and_play(
                    lexicon.apply(sec.title),
                    lang_code=lang_code,
                    voice=voice,
                    display=display,
                    speed=TITLE_SPEED,
                    device=device,
                )
                time.sleep(TITLE_PAUSE_SECONDS)

            body = lexicon.apply(get_section_body(sec))
            if body:
                synthesize_and_play(
                    body,
                    lang_code=lang_code,
                    voice=voice,
                    display=display,
                    speed=speed,
                    device=device,
                )
        except (PlayerNotFoundError, GpuMemoryError):
            raise
        except TTSError as exc:
            failed += 1
            print(
                f"Warning: Failed to read section {sec.index}: {exc}",
                file=sys.stderr,
            )
    return failed


def export_sections(
    sections: list,
    total: int,
    output_dir: str,
    lang_code: str,
    voice: str | None,
    speed: float,
    device: str,
    audio_format: str,
    mp3_quality: int,
    lexicon: PronunciationDictionary,
) -> int:
    done = skipped = failed = 0
    for sec in sections:
        name = sanitize_filename(sec.title)
        filename = f"{sec.index:02d}_{name}.{audio_format}"
        output_filepath = os.path.join(output_dir, filename)

        if os.path.exists(output_filepath):
            print(
                f"[{sec.index}/{total}] Skipping (already completed): "
                f"{filename}"
            )
            skipped += 1
            continue

        body = get_section_body(sec)
        if not body:
            print(
                f"[{sec.index}/{total}] Skipping empty or too short "
                f"section: {filename}"
            )
            skipped += 1
            continue

        print(f"[{sec.index}/{total}] Synthesizing section: {filename}")
        try:
            synthesize_epub_section_to_file(
                lexicon.apply(sec.title) if sec.has_heading else "",
                lexicon.apply(body),
                output_filepath,
                lang_code=lang_code,
                voice=voice,
                speed=speed,
                device=device,
                audio_format=audio_format,
                mp3_quality=mp3_quality,
            )
        except GpuMemoryError:
            raise
        except TTSError as exc:
            failed += 1
            print(
                f"Warning: Failed to process section {sec.index}: {exc}",
                file=sys.stderr,
            )
            continue
        done += 1
        print(f"    -> Completed successfully: {filename}")

    print(
        f"\n[Book Processing] Finished: {done} exported, {skipped} skipped, "
        f"{failed} failed. Files are in '{output_dir}/'."
    )
    return failed


def process_book_file(
    file_path: str,
    is_epub: bool,
    lang_code: str,
    voice: str | None,
    output_arg: str | None,
    section_list: bool,
    section_start: str | None,
    display: bool,
    speed: float = 1.0,
    device: str = "cpu",
    audio_format: str = "flac",
    mp3_quality: int = DEFAULT_MP3_QUALITY,
    lexicon: PronunciationDictionary | None = None,
) -> int:
    if is_epub and not EPUB_SUPPORT:
        raise TTSError(
            "'ebooklib' or 'beautifulsoup4' is not installed. "
            "Run 'pip install ebooklib beautifulsoup4'"
        )
    if not os.path.exists(file_path):
        file_type = "EPUB" if is_epub else "TXT"
        raise TTSError(f"{file_type} file '{file_path}' not found.")

    if lexicon is None:
        lexicon = PronunciationDictionary({})

    try:
        sections = extract_sections(file_path) if is_epub else extract_txt_sections(file_path)
    except Exception as exc:
        raise TTSError(f"Cannot read file '{file_path}': {exc}") from exc

    if section_list:
        print(f"\nSections/Blocks found in '{file_path}':")
        print("-" * 60)
        for sec in sections:
            print(f"  [{sec.index}] {sec.title}")
        print()
        return 0

    start_idx = find_start_index(sections, section_start)
    selected = [sec for sec in sections if sec.index >= start_idx]
    total = len(sections)

    if output_arg is None:
        print(
            f"[Streaming Mode] Reading document starting from section "
            f"{start_idx}/{total}...\n"
        )
        failed = stream_sections(
            selected, total, lang_code, voice, display, speed, device,
            lexicon,
        )
        print("\n[Streaming] Complete.")
        return failed

    if output_arg == "":
        output_dir = os.path.splitext(os.path.basename(file_path))[0]
    else:
        output_dir = output_arg

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        raise TTSError(
            f"Cannot create directory '{output_dir}': {exc}"
        ) from exc
    print(f"[Export Mode] Output directory ready: '{output_dir}'")
    print(
        f"[Export Mode] Processing {len(selected)} sections starting "
        f"from {start_idx}...\n"
    )
    return export_sections(
        selected, total, output_dir, lang_code, voice, speed, device,
        audio_format, mp3_quality, lexicon,
    )


def run_voice_tests(lang_code: str, device: str = "cpu") -> int:
    code = lang_code.lower()
    if code not in SUPPORTED_LANGUAGES:
        raise TTSError(
            f"Unsupported language code '{lang_code}' for testing. "
            "Use --list-languages to view available options."
        )

    voices = ALL_VOICES.get(ISO_TO_VOICE_KEY.get(code, ""), [])
    if not voices:
        raise TTSError(f"No voices found for language '{code}'.")

    test_text = TEST_TEXTS.get(code, TEST_TEXTS["en"])
    print(f"Testing {code} voices with: {test_text}")
    print("-" * 60)

    failed = 0
    for number, voice in enumerate(voices, 1):
        print(f"[{number}/{len(voices)}] Playing voice -> {voice}...")
        try:
            synthesize_and_play(
                test_text, lang_code=code, voice=voice, device=device
            )
        except (PlayerNotFoundError, GpuMemoryError):
            raise
        except TTSError as exc:
            failed += 1
            print(f"Warning: voice '{voice}' failed: {exc}", file=sys.stderr)

    if failed:
        print(f"\n[TEST MODE] {failed} of {len(voices)} voices failed.")
    else:
        print("\n[TEST MODE] All voices tested successfully.")
    return failed


# --------------------------------------------------------------------------
# Command line interface
# --------------------------------------------------------------------------

def positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a number")
    if number <= 0:
        raise argparse.ArgumentTypeError("the value must be greater than 0")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Kokoro-82M Local Neural Text-to-Speech client. "
            "Converts text, TXT, or EPUB books into natural speech "
            "using local inference."
        ),
        epilog=(
            "Examples:\n"
            '  python kokoro_tts.py "Hello, this is a test."\n'
            '  python kokoro_tts.py "Hello" --output greeting.flac\n'
            "  python kokoro_tts.py --epub book.epub --analyze\n"
            "  python kokoro_tts.py --txt document.txt --output\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "text",
        nargs="?",
        help=(
            "The text to convert to speech. If omitted, reads from "
            "standard input, EPUB, or TXT."
        ),
    )
    parser.add_argument(
        "--lang",
        default="es",
        type=str.lower,
        choices=sorted(SUPPORTED_LANGUAGES),
        help="Language code. Default: 'es'.",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help=(
            "Neural voice ID to use. If omitted, uses the first verified "
            "voice for the selected language."
        ),
    )
    parser.add_argument(
        "--speed",
        type=positive_float,
        default=1.0,
        help="Speech speed multiplier (default: 1.0).",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help=(
            "Where to run the model: 'cpu' (default), 'cuda' (GPU) or "
            "'auto' (GPU if it has free memory, otherwise CPU)."
        ),
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help=(
            "Display text on screen and highlight words in real-time as "
            "they are spoken."
        ),
    )
    parser.add_argument(
        "--epub",
        metavar="FILE.epub",
        help="Path to an EPUB book.",
    )
    parser.add_argument(
        "--txt",
        metavar="FILE.txt",
        help="Path to a TXT text document.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help=(
            "Analyze EPUB or TXT text to extract unknown/out-of-dictionary words "
            "into a .dict file."
        ),
    )
    parser.add_argument(
        "--output",
        nargs="?",
        const="",
        metavar="TARGET",
        help=(
            "Export EPUB/TXT to directory OR write plain text synthesis directly "
            "to audio file. For plain text, a target filename is mandatory."
        ),
    )
    parser.add_argument(
        "--format",
        choices=AUDIO_FORMATS,
        default="flac",
        help=(
            "Audio format of the files exported with --output: 'flac' "
            "(default) or 'mp3' (variable bitrate, needs ffmpeg with "
            "libmp3lame)."
        ),
    )
    parser.add_argument(
        "--mp3-quality",
        type=int,
        choices=range(10),
        metavar="0-9",
        default=DEFAULT_MP3_QUALITY,
        help=(
            "Variable-bitrate quality of MP3 files: 0 is the best and "
            f"largest, 9 the smallest (default: {DEFAULT_MP3_QUALITY})."
        ),
    )
    parser.add_argument(
        "--dictionary",
        metavar="FILE",
        help=(
            "Pronunciation dictionary: a UTF-8 text file with one "
            "'word|replacement' per line (e.g. 'mr|[mˈɪstəɹ]'). "
            "In combination with --analyze, specifies the output .dict file name."
        ),
    )
    parser.add_argument(
        "--dictionary-ignore-case",
        action="store_true",
        help="Match words of --dictionary regardless of capitalization.",
    )
    parser.add_argument(
        "--section-list",
        action="store_true",
        help="List all sections and titles in the EPUB/TXT and exit.",
    )
    parser.add_argument(
        "--section-start",
        metavar="SECTION",
        help=(
            "Start reading or exporting from the specified section index "
            "or title substring."
        ),
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="Display all supported ISO language codes and exit.",
    )
    parser.add_argument(
        "--list-voices",
        nargs="?",
        const="all",
        metavar="LANG",
        help=(
            "Display available neural voices. Optionally pass an ISO code "
            "to filter."
        ),
    )
    parser.add_argument(
        "--test",
        metavar="LANG",
        help=(
            "Test all available voices for the specified ISO language "
            "using preset presentation text."
        ),
    )
    return parser


def prepare_synthesis(
    args: argparse.Namespace,
    needs_player: bool,
    lang_code: str,
    needs_mp3: bool = False,
) -> str:
    validate_voice(lang_code, args.voice)
    if needs_player:
        require_ffplay()
    if needs_mp3:
        require_mp3_encoder()
    return resolve_device(args.device)


def load_lexicon(args: argparse.Namespace) -> PronunciationDictionary:
    if not args.dictionary or args.analyze:
        return PronunciationDictionary({})
    lexicon = load_dictionary(args.dictionary, args.dictionary_ignore_case)
    print(
        f"[INFO] Loaded {len(lexicon)} pronunciation entries from "
        f"'{args.dictionary}'."
    )
    return lexicon


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.list_languages:
        show_languages()
        return 0

    if args.list_voices is not None:
        show_voices(lang_filter=args.list_voices)
        return 0

    if args.test:
        lang = args.test.lower()
        if lang not in SUPPORTED_LANGUAGES:
            raise TTSError(
                f"Unsupported language code '{args.test}' for testing. "
                "Use --list-languages to view available options."
            )
        device = prepare_synthesis(args, needs_player=True, lang_code=lang)
        return 1 if run_voice_tests(lang, device=device) else 0

    if (args.section_list or args.section_start or args.analyze) and not (args.epub or args.txt):
        parser.error(
            "--section-list, --section-start, and --analyze require --epub or --txt."
        )

    if sum(bool(x) for x in (args.epub, args.txt, args.text)) > 1:
        parser.error("Please specify only one source: positional text, --epub, or --txt.")

    target_file = args.epub or args.txt
    is_epub = bool(args.epub)

    # Document (EPUB/TXT) + --analyze
    if target_file and args.analyze:
        if args.dictionary:
            dict_path = args.dictionary
        else:
            base_name = os.path.splitext(target_file)[0]
            dict_path = f"{base_name}.dict"

        device = prepare_synthesis(args, needs_player=False, lang_code=args.lang)
        return analyze_book_words(
            file_path=target_file,
            lang_code=args.lang,
            dict_output_path=dict_path,
            is_epub=is_epub,
            device=device,
        )

    # Document (EPUB/TXT) Normal workflow
    if target_file:
        streaming = args.output is None and not args.section_list
        exporting_mp3 = args.output is not None and args.format == "mp3"
        device = "cpu"
        lexicon = PronunciationDictionary({})
        if not args.section_list:
            lexicon = load_lexicon(args)
            device = prepare_synthesis(
                args,
                needs_player=streaming,
                lang_code=args.lang,
                needs_mp3=exporting_mp3,
            )
        failed = process_book_file(
            file_path=target_file,
            is_epub=is_epub,
            lang_code=args.lang,
            voice=args.voice,
            output_arg=args.output,
            section_list=args.section_list,
            section_start=args.section_start,
            display=args.display,
            speed=args.speed,
            device=device,
            audio_format=args.format,
            mp3_quality=args.mp3_quality,
            lexicon=lexicon,
        )
        return 1 if failed else 0

    # Plain Text workflow
    if args.text:
        text_to_speak = args.text
    elif not sys.stdin.isatty():
        text_to_speak = sys.stdin.read().strip()
    else:
        parser.print_help()
        return 1

    if not text_to_speak:
        raise TTSError("There is no text to synthesize (empty input).")

    if args.output is not None:
        if args.output == "":
            raise TTSError(
                "You must specify an output filename for --output "
                "when processing plain text."
            )
        output_file = args.output
        needs_mp3 = args.format == "mp3" or output_file.lower().endswith(".mp3")
        audio_fmt = "mp3" if needs_mp3 else args.format

        lexicon = load_lexicon(args)
        device = prepare_synthesis(
            args,
            needs_player=False,
            lang_code=args.lang,
            needs_mp3=needs_mp3,
        )
        synthesize_text_to_file(
            lexicon.apply(text_to_speak),
            output_filepath=output_file,
            lang_code=args.lang,
            voice=args.voice,
            speed=args.speed,
            device=device,
            audio_format=audio_fmt,
            mp3_quality=args.mp3_quality,
        )
        return 0

    lexicon = load_lexicon(args)
    device = prepare_synthesis(args, needs_player=True, lang_code=args.lang)
    synthesize_and_play(
        lexicon.apply(text_to_speak),
        lang_code=args.lang,
        voice=args.voice,
        display=args.display,
        speed=args.speed,
        device=device,
    )
    return 0


def main(argv: list | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args, parser)
    except GpuMemoryError as exc:
        print(f"Warning: {exc}", file=sys.stderr)
        return EXIT_GPU_BUSY
    except TTSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "\n\n[INFO] Execution aborted by user (Ctrl+C). "
            "Exiting cleanly."
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
