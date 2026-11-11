"""Supported languages, voices and their listings."""

from __future__ import annotations

import os
import sys

from bts.errors import TTSError

# Mapping ISO language codes to Kokoro internal pipeline codes.
SUPPORTED_LANGUAGES = {
    "es": {"internal": "e", "name": "Spanish (Español)"},
    "en": {"internal": "b", "name": "English (Defaults to British)"},
    "uk": {"internal": "b", "name": "English - British (UK)"},
    "us": {"internal": "a", "name": "English - American (US)"},
    "fr": {"internal": "f", "name": "French (Français)"},
    "it": {"internal": "i", "name": "Italian (Italiano)"},
    "ja": {"internal": "j", "name": "Japanese (日本語)"},
    "zh": {"internal": "z", "name": "Mandarin Chinese (中文)"},
    "pt": {"internal": "p", "name": "Portuguese (Português)"},
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
    "zh": "zh (Mandarin Chinese)",
    "pt": "pt (Portuguese)",
}


KNOWN_VOICES = frozenset(
    voice for voices in ALL_VOICES.values() for voice in voices
)


_ENGLISH_TEST_TEXT = (
    "Hello, I am an AI-synthesized voice, pleased to greet you. "
    "I hope you make good use of me from now on."
)


# Presentation text used by ``--test`` for each language.
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
        "Olá, sou uma voz sintetizada por IA, prazer em conhecê-lo. "
        "Espero que faça bom uso de mim a partir de agora."
    ),
}


def get_default_voice(lang_code: str) -> str:
    """Return the first verified voice for an ISO language code.

    Args:
        lang_code: ISO language code such as ``"es"`` or ``"us"``.

    Returns:
        The voice identifier. Falls back to ``"ef_dora"`` for unknown codes.
    """
    voice_key = ISO_TO_VOICE_KEY.get(lang_code.lower())
    if voice_key and ALL_VOICES.get(voice_key):
        return ALL_VOICES[voice_key][0]
    return "ef_dora"


def validate_voice(lang_code: str, voice: str | None) -> None:
    """Check that a user-selected voice exists.

    A path to a custom ``.pt`` voice file is also accepted. A voice that
    belongs to a different language than ``lang_code`` is allowed, but a
    warning is printed because the pronunciation will probably be poor.

    Args:
        lang_code: ISO language code selected by the user.
        voice: Voice identifier or path, or None to use the default.

    Raises:
        TTSError: If the voice is neither a known voice nor an existing file.
    """
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
    """Display the supported ISO language codes and descriptions."""
    print("\nSupported ISO Languages:")
    print("-" * 50)
    for code, info in SUPPORTED_LANGUAGES.items():
        print(f"  [{code}] -> {info['name']}")
    print()


def show_voices(lang_filter: str = "all") -> None:
    """Display the available voices, optionally for a single language.

    Args:
        lang_filter: ISO language code, or ``"all"`` to list every voice.

    Raises:
        TTSError: If ``lang_filter`` is not a supported language code.
    """
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
