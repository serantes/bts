"""Speech generation, playback and export to audio files."""

from __future__ import annotations

import contextlib
import math
import os
import re
import sys
from typing import TYPE_CHECKING, Callable

import numpy as np

from bts.audio import (
    AudioPlayer,
    KaraokeDisplay,
    open_audio_writer,
    to_float32,
)
from bts.constants import (
    DEFAULT_MAX_PAUSE_SECONDS,
    DEFAULT_MP3_QUALITY,
    ENGLISH_LANG_CODES,
    PAUSE_MARKERS,
    PHONEME_CHUNK_CHARS,
    PHONEME_MARKUP_RE,
    SAMPLE_RATE,
    TITLE_PAUSE_SECONDS,
    TITLE_SPEED,
)
from bts.device import get_pipeline, get_torch, gpu_oom_message, is_cuda_oom
from bts.errors import GpuMemoryError, PlayerNotFoundError, TTSError, warn_once
from bts.languages import (
    ALL_VOICES,
    ISO_TO_VOICE_KEY,
    SUPPORTED_LANGUAGES,
    TEST_TEXTS,
    get_default_voice,
)
from bts.numbers import remove_ignored_strings
from bts.phonemes import _split_phonemes, chunk_phonemes, strip_phoneme_markup
from bts.textio import split_into_chunks

if TYPE_CHECKING:
    from kokoro import KPipeline


_PAUSE_MARKER_RE = re.compile(
    "|".join(
        re.escape(marker)
        for marker in sorted(PAUSE_MARKERS, key=len, reverse=True)
    )
)


class SynthesisCancelled(Exception):
    """Raised when a daemon request is stopped between audio chunks."""


def _checkpoint(checkpoint: Callable[[], None] | None) -> None:
    if checkpoint is not None:
        checkpoint()


def _generate_with_phonemes(
    pipeline: KPipeline, text: str, voice: str, speed: float
):
    """Speak text with phoneme markup in a language other than English.

    Kokoro only understands ``[word](/phonemes/)`` in English. For the
    other languages, the text around each marker is phonemized with the
    G2P of the pipeline, the given phonemes are inserted, and the whole
    chunk is synthesized at once with ``generate_from_tokens``, so the
    intonation stays continuous. If that is not possible, the markers are
    replaced by their plain word and a warning is shown.

    Yields:
        ``(graphemes, audio)`` pairs.
    """
    can_inject = hasattr(pipeline, "generate_from_tokens")
    if not can_inject:
        warn_once(
            "this version of kokoro cannot synthesize phonemes directly; "
            "the IPA entries are read as plain words."
        )
    for line in re.split(r"\n+", text):
        if not line.strip():
            continue
        for chunk in split_into_chunks(line, PHONEME_CHUNK_CHARS):
            phonemes = None
            if can_inject and PHONEME_MARKUP_RE.search(chunk):
                phonemes = chunk_phonemes(pipeline, chunk)
            plain = strip_phoneme_markup(chunk)
            if phonemes is None:
                for graphemes, _, audio in pipeline(
                    plain, voice=voice, speed=speed, split_pattern=r"\n+"
                ):
                    if audio is not None:
                        yield graphemes, audio
                continue
            for sub_phonemes in _split_phonemes(phonemes):
                for _, _, audio in pipeline.generate_from_tokens(
                    sub_phonemes, voice=voice, speed=speed
                ):
                    if audio is not None:
                        yield plain, audio


def _generate_speech_text(
    pipeline: KPipeline,
    internal_lang: str,
    text: str,
    voice: str,
    speed: float,
):
    """Generate speech for a marker-free text segment."""
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


def generate_speech(
    pipeline: KPipeline,
    internal_lang: str,
    text: str,
    voice: str,
    speed: float,
    max_pause_seconds: float = DEFAULT_MAX_PAUSE_SECONDS,
):
    """Generate speech and insert silence for configured text markers.

    Literal markers and their pause durations are configured in
    ``bts.constants.PAUSE_MARKERS`` and capped by ``max_pause_seconds``.
    Consecutive markers separated only by whitespace become one pause. The
    markers themselves are not spoken.
    """
    if not math.isfinite(max_pause_seconds) or max_pause_seconds < 0:
        raise ValueError("max_pause_seconds must be finite and non-negative")

    text = remove_ignored_strings(text)
    matches = list(_PAUSE_MARKER_RE.finditer(text))
    position = 0
    index = 0
    while index < len(matches):
        match = matches[index]
        speech_text = text[position:match.start()]
        if speech_text.strip():
            yield from _generate_speech_text(
                pipeline, internal_lang, speech_text, voice, speed
            )

        end_index = index + 1
        while (
            end_index < len(matches)
            and not text[
                matches[end_index - 1].end():matches[end_index].start()
            ].strip()
        ):
            end_index += 1
        marker_pause = max(
            PAUSE_MARKERS[item.group()]
            for item in matches[index:end_index]
        )
        pause_seconds = min(marker_pause, max_pause_seconds)
        if pause_seconds > 0:
            yield "", np.zeros(
                int(SAMPLE_RATE * pause_seconds), dtype=np.float32
            )
        position = matches[end_index - 1].end()
        index = end_index

    speech_text = text[position:]
    if speech_text.strip():
        yield from _generate_speech_text(
            pipeline, internal_lang, speech_text, voice, speed
        )


def synthesize_and_play(
    text: str,
    lang_code: str = "es",
    voice: str | None = None,
    display: bool = False,
    speed: float = 1.0,
    device: str = "cpu",
    title: str = "",
    checkpoint: Callable[[], None] | None = None,
    max_pause_seconds: float = DEFAULT_MAX_PAUSE_SECONDS,
    title_speed_factor: float = TITLE_SPEED,
) -> None:
    """Generate speech from text and stream it to ``ffplay``.

    Args:
        text: Text to speak.
        lang_code: ISO language code (see ``SUPPORTED_LANGUAGES``).
        voice: Voice identifier; the language default if omitted.
        display: If True, show the text with a moving word highlight.
        speed: Speech speed multiplier.
        device: ``"cpu"`` or ``"cuda"``.
        title: Optional heading read first, slower by default, followed by
            a pause, in the same audio stream as the text.
        title_speed_factor: Multiplier applied to speed for the title.

    Raises:
        PlayerNotFoundError: If ``ffplay`` is not installed.
        GpuMemoryError: If the GPU runs out of memory.
        TTSError: If generation or playback fails.
    """
    lang_info = SUPPORTED_LANGUAGES.get(lang_code.lower())
    if not lang_info:
        raise TTSError(
            f"Unsupported language code '{lang_code}'. "
            "Use --list-languages to view available options."
        )

    selected_voice = voice or get_default_voice(lang_code)
    pipeline = get_pipeline(lang_info["internal"], device)
    internal = lang_info["internal"]

    def segments():
        if title:
            yield from generate_speech(
                pipeline, internal, title, selected_voice,
                speed * title_speed_factor,
                max_pause_seconds,
            )
            yield "", np.zeros(
                int(SAMPLE_RATE * TITLE_PAUSE_SECONDS), dtype=np.float32
            )
        if text:
            yield from generate_speech(
                pipeline, internal, text, selected_voice, speed,
                max_pause_seconds,
            )

    generator = segments()
    karaoke = KaraokeDisplay() if display else None
    completed = False
    try:
        with AudioPlayer() as player:
            for graphemes, audio in generator:
                _checkpoint(checkpoint)
                samples = to_float32(audio)
                player.write(samples)
                if karaoke is not None and graphemes.strip():
                    karaoke.submit(
                        graphemes.strip(), len(samples) / SAMPLE_RATE
                    )
        completed = True
    except (TTSError, SynthesisCancelled):
        raise
    except Exception as exc:
        if is_cuda_oom(exc):
            get_torch().cuda.empty_cache()
            raise GpuMemoryError(gpu_oom_message()) from exc
        raise TTSError(
            f"Error during audio generation or playback: {exc}"
        ) from exc
    finally:
        if karaoke is not None:
            karaoke.close(abort=not completed)


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
    checkpoint: Callable[[], None] | None = None,
    max_pause_seconds: float = DEFAULT_MAX_PAUSE_SECONDS,
) -> None:
    """Synthesize a section (title, pause, body) into a FLAC or MP3 file.

    The audio is written incrementally to a temporary file that is renamed
    when complete, so an interrupted run never leaves a half-written file
    that would be skipped on the next run.

    Args:
        title_text: Heading to read before the body; empty to skip it.
        body_text: Section text.
        output_filepath: Destination path.
        lang_code: ISO language code.
        voice: Voice identifier; the language default if omitted.
        speed: Speech speed multiplier for the body text.
        device: ``"cpu"`` or ``"cuda"``.
        audio_format: ``"flac"`` or ``"mp3"``.
        mp3_quality: MP3 variable-bitrate quality, from 0 (best, largest)
            to 9 (smallest). Ignored for FLAC.

    Raises:
        GpuMemoryError: If the GPU runs out of memory.
        TTSError: If no audio could be generated or the file cannot be
            written.
    """
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
                    speed * TITLE_SPEED,
                    max_pause_seconds,
                ):
                    _checkpoint(checkpoint)
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
                _checkpoint(checkpoint)
                for _, audio in generate_speech(
                    pipeline,
                    lang_info["internal"],
                    chunk,
                    selected_voice,
                    speed,
                    max_pause_seconds,
                ):
                    _checkpoint(checkpoint)
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
    except (TTSError, SynthesisCancelled):
        raise
    except Exception as exc:
        if is_cuda_oom(exc):
            get_torch().cuda.empty_cache()
            raise GpuMemoryError(gpu_oom_message()) from exc
        raise TTSError(
            f"Error generating audio for file '{output_filepath}': {exc}"
        ) from exc
    finally:
        if not finished:
            print()
            with contextlib.suppress(OSError):
                os.remove(temp_filepath)


def synthesize_text_to_file(
    text: str,
    output_filepath: str,
    lang_code: str = "es",
    voice: str | None = None,
    speed: float = 1.0,
    device: str = "cpu",
    audio_format: str = "flac",
    mp3_quality: int = DEFAULT_MP3_QUALITY,
    checkpoint: Callable[[], None] | None = None,
    max_pause_seconds: float = DEFAULT_MAX_PAUSE_SECONDS,
) -> None:
    """Synthesize a text into a single FLAC or MP3 file.

    An existing file is replaced, but only once the new one is complete.

    Args:
        text: Text to speak. Every line is read as a paragraph.
        output_filepath: Destination path, including the extension.
        lang_code: ISO language code.
        voice: Voice identifier; the language default if omitted.
        speed: Speech speed multiplier.
        device: ``"cpu"`` or ``"cuda"``.
        audio_format: ``"flac"`` or ``"mp3"``.
        mp3_quality: MP3 variable-bitrate quality (0 best, 9 smallest).

    Raises:
        GpuMemoryError: If the GPU runs out of memory.
        TTSError: If the audio cannot be generated or written.
    """
    directory = os.path.dirname(output_filepath)
    if directory:
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as exc:
            raise TTSError(
                f"Cannot create directory '{directory}': {exc}"
            ) from exc
    synthesize_epub_section_to_file(
        "",
        text,
        output_filepath,
        lang_code=lang_code,
        voice=voice,
        speed=speed,
        device=device,
        audio_format=audio_format,
        mp3_quality=mp3_quality,
        checkpoint=checkpoint,
        max_pause_seconds=max_pause_seconds,
    )
    print(f"Audio saved to '{output_filepath}'.")


def run_voice_tests(lang_code: str, device: str = "cpu") -> int:
    """Play a preset text with every voice of a language.

    Args:
        lang_code: ISO language code.
        device: ``"cpu"`` or ``"cuda"``.

    Returns:
        The number of voices that failed.

    Raises:
        TTSError: If the language is unsupported or has no voices.
    """
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
