"""Kokoro's inline phoneme markup and phoneme helpers."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from bts.constants import MAX_PHONEMES, PHONEME_FORBIDDEN_RE, PHONEME_MARKUP_RE
from bts.errors import warn_once

if TYPE_CHECKING:
    from kokoro import KPipeline


def parse_phonemes(spoken: str) -> str | None:
    """Return the phonemes of a replacement written as ``[phonemes]``.

    Args:
        spoken: The replacement text of a dictionary entry.

    Returns:
        The phonemes between the brackets (possibly empty), or None if the
        replacement is plain text.

    Raises:
        ValueError: If the phonemes contain any of ``[ ] ( ) /``.
    """
    text = spoken.strip()
    if len(text) < 2 or text[0] != "[" or text[-1] != "]":
        return None
    phonemes = text[1:-1].strip()
    if PHONEME_FORBIDDEN_RE.search(phonemes):
        raise ValueError("phonemes cannot contain any of [ ] ( ) /")
    return phonemes


def phoneme_markup(word: str, phonemes: str) -> str:
    """Return Kokoro's inline syntax ``[label](/phonemes/)`` for a word.

    The label is made of the letters and digits of the word, because Kokoro
    expects a single token there.
    """
    label = re.sub(r"[\W_]+", "", word) or "x"
    return f"[{label}](/{phonemes}/)"


def strip_phoneme_markup(text: str) -> str:
    """Replace every ``[word](/phonemes/)`` by ``word``."""
    return PHONEME_MARKUP_RE.sub(r"\1", text)


def text_phonemes(pipeline: KPipeline, text: str) -> str:
    """Convert plain text to phonemes with the G2P of the pipeline."""
    if not text.strip():
        return ""
    result = pipeline.g2p(text)
    phonemes = result[0] if isinstance(result, tuple) else result
    return (phonemes or "").strip()


def join_phonemes(parts: list) -> str:
    """Join phoneme strings with spaces, but not before punctuation."""
    joined = ""
    for part in parts:
        if not part:
            continue
        if joined and part[0] not in ",.;:!?…":
            joined += " "
        joined += part
    return joined


def chunk_phonemes(pipeline: KPipeline, chunk: str) -> str | None:
    """Build the phonemes of a chunk that contains phoneme markup.

    Args:
        pipeline: The Kokoro pipeline of the language.
        chunk: Text with ``[word](/phonemes/)`` markup.

    Returns:
        The phoneme string, or None if it cannot be built or is too long.
    """
    try:
        parts = []
        position = 0
        for match in PHONEME_MARKUP_RE.finditer(chunk):
            before = chunk[position:match.start()]
            parts.append(text_phonemes(pipeline, before))
            parts.append(match.group(2).strip())
            position = match.end()
        parts.append(text_phonemes(pipeline, chunk[position:]))
    except Exception as exc:
        warn_once(
            f"could not phonemize the text around the IPA entries ({exc}); "
            "they are read as plain words."
        )
        return None

    phonemes = join_phonemes(parts)
    if len(phonemes) > MAX_PHONEMES:
        warn_once(
            "a text chunk with IPA entries is too long; its IPA entries "
            "are read as plain words."
        )
        return None
    return phonemes
