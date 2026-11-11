"""Text splitting, text files and audio output names."""

from __future__ import annotations

import os
import re

from bts.constants import AUDIO_FORMATS, MAX_CHUNK_CHARS
from bts.errors import TTSError


def split_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list:
    """Split text into chunks made of whole sentences.

    Sentences are merged until a chunk reaches ``max_chars``. A single
    sentence longer than that is kept whole. Line breaks are never merged:
    every line starts a new chunk. Latin and CJK sentence terminators are
    recognized.

    Args:
        text: Text to split.
        max_chars: Target maximum chunk length in characters.

    Returns:
        The list of non-empty chunks.
    """
    chunks: list = []
    for paragraph in re.split(r"\n+", text):
        current = ""
        sentences = re.split(r"(?<=[.!?])\s+|(?<=[。！？])\s*", paragraph)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if current and len(current) + 1 + len(sentence) > max_chars:
                chunks.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            chunks.append(current)
    return chunks


def read_text_file(path: str) -> str:
    """Read a UTF-8 text file.

    Args:
        path: Path to the text file.

    Returns:
        The text, without leading or trailing whitespace. Line breaks
        inside the text are kept, because every line is read as a separate
        paragraph.

    Raises:
        TTSError: If the file is missing, empty, unreadable or not UTF-8.
    """
    if not os.path.isfile(path):
        raise TTSError(f"Text file '{path}' not found.")
    try:
        with open(path, encoding="utf-8-sig") as handle:
            text = handle.read().strip()
    except UnicodeDecodeError as exc:
        raise TTSError(f"'{path}' is not a valid UTF-8 file: {exc}") from exc
    except OSError as exc:
        raise TTSError(f"Cannot read '{path}': {exc}") from exc
    if not text:
        raise TTSError(f"Text file '{path}' is empty.")
    return text


def resolve_audio_output(
    output_arg: str, format_arg: str | None, default_name: str | None
) -> tuple:
    """Work out the path and the format of the audio file to create.

    If the name ends in ``.flac`` or ``.mp3``, the extension selects the
    format. Otherwise the extension of ``--format`` (``flac`` by default)
    is added to the name.

    Args:
        output_arg: Value of ``--output``; an empty string if the flag was
            given without a name.
        format_arg: Value of ``--format``, or None if it was not given.
        default_name: Name, without extension, to use when ``--output`` has
            no name (the one of the ``--txt`` file), or None if there is no
            such name.

    Returns:
        A tuple ``(path, audio_format)``.

    Raises:
        TTSError: If there is no file name, if ``--format`` contradicts the
            extension of the name, or if the path is a directory.
    """
    name = output_arg or default_name
    if not name:
        raise TTSError(
            "--output needs a file name when the text does not come from "
            "--txt, for example: --output reading.mp3"
        )
    extension = os.path.splitext(name)[1].lower().lstrip(".")
    if extension in AUDIO_FORMATS:
        if format_arg and format_arg != extension:
            raise TTSError(
                f"--format {format_arg} contradicts the extension of "
                f"'{name}'."
            )
        audio_format, path = extension, name
    else:
        audio_format = format_arg or "flac"
        path = f"{name}.{audio_format}"
    if os.path.isdir(path):
        raise TTSError(f"'{path}' is a directory, not a file name.")
    return path, audio_format
