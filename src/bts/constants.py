"""Constants shared by the whole application."""

from __future__ import annotations

import re

AUDIO_FORMATS = ("flac", "mp3")  # Formats available for EPUB export.

DEFAULT_MAX_PAUSE_SECONDS = 0.5
DEFAULT_MP3_QUALITY = 4  # LAME VBR quality: 0 = best/largest, 9 = smallest.

DICTIONARY_SEPARATOR = "|"  # Separates the two columns of the dictionary.

ENGLISH_LANG_CODES = ("a", "b")  # Kokoro pipelines that use English G2P.
EXIT_GPU_BUSY = 0  # Exit status when the GPU has no free memory.

GPU_MIN_FREE_MIB = 1024  # Free VRAM required to use the GPU (Kokoro ~830).

IGNORED_STRINGS = ["\u2060"]  # Removed before text is prepared or synthesized.

MAX_CHUNK_CHARS = 400  # Target size of text chunks used when exporting.
MAX_PHONEMES = 510  # Longest phoneme string that Kokoro accepts at once.
MIN_SECTION_CHARS = 10  # Shorter sections are considered empty.

PAUSE_MARKERS = {
    "—": DEFAULT_MAX_PAUSE_SECONDS,
    "!": DEFAULT_MAX_PAUSE_SECONDS,
    "?": DEFAULT_MAX_PAUSE_SECONDS,
}
PHONEME_CHUNK_CHARS = 300  # Text chunk size when phonemes are injected.
PHONEME_FORBIDDEN_RE = re.compile(r"[\[\]()/\n]")
PHONEME_MARKUP_RE = re.compile(r"\[(\w+)\]\(/([^/()\[\]\n]*)/\)")

REPO_ID = "hexgrad/Kokoro-82M"

SAMPLE_RATE = 24000  # Output sample rate of Kokoro-82M, in Hz.

TITLE_PAUSE_SECONDS = 0.8  # Silence between a section title and its body.
TITLE_SPEED = 0.8  # Section titles are read slower than the body text.
