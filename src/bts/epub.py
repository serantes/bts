"""EPUB reading: sections, streaming and export."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass

from bts.constants import DEFAULT_MP3_QUALITY, MIN_SECTION_CHARS
from bts.dictionary import PronunciationDictionary
from bts.errors import GpuMemoryError, PlayerNotFoundError, TTSError
from bts.synthesis import synthesize_and_play, synthesize_epub_section_to_file

# Optional imports for EPUB processing.
try:
    import ebooklib
    from bs4 import BeautifulSoup, NavigableString, Tag
    from ebooklib import epub

    EPUB_SUPPORT = True
except ImportError:
    EPUB_SUPPORT = False


# HTML elements that start a new block of text or a separate token (a
# footnote number, for instance). A space is placed around them when the
# text of an EPUB is extracted. Inline formatting (<a>, <span>, <i>, <b>,
# ...) is not listed: it must not add spaces inside a word, as in
# "<a>A</a>ndrew".
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


@dataclass
class Section:
    """A document of an EPUB book.

    Attributes:
        index: 1-based position of the section in the book.
        title: Heading text, or ``"Section N"`` if the document has none.
        has_heading: True if ``title`` comes from a real heading and can be
            read aloud.
        soup: Parsed HTML content of the section.
    """

    index: int
    title: str
    has_heading: bool
    soup: BeautifulSoup


def sanitize_filename(name: str) -> str:
    """Turn a section title into a safe file name (at most 50 characters).

    Args:
        name: Section or chapter title.

    Returns:
        The sanitized name, or ``"section"`` if nothing usable remains.
    """
    cleaned = re.sub(r'[\\/*?:"<>|]', "", name)
    cleaned = re.sub(r"\s+", "_", cleaned).strip("_")
    return cleaned[:50] if cleaned else "section"


def _collect_text(node: Tag, parts: list) -> None:
    """Append the text found under ``node`` to ``parts``, in order."""
    for child in node.children:
        if isinstance(child, Tag):
            local_name = (child.name or "").rsplit(":", 1)[-1].lower()
            separate = local_name in TEXT_BREAK_TAGS
            if separate:
                parts.append(" ")
            _collect_text(child, parts)
            if separate:
                parts.append(" ")
        elif type(child) is NavigableString:  # not comments, scripts, ...
            parts.append(str(child))


def get_plain_text(node: Tag) -> str:
    """Return the text of an HTML node with normalized whitespace.

    Block-level elements (paragraphs, headings, list items, line breaks,
    table cells, ...) are separated by a space, but inline formatting adds
    nothing: ``<a>A</a>ndrew`` gives ``Andrew``, not ``A ndrew``. Runs of
    whitespace, including line breaks and non-breaking spaces, become a
    single space.

    Args:
        node: A BeautifulSoup document or tag.

    Returns:
        The text, without leading or trailing whitespace.
    """
    parts: list = []
    _collect_text(node, parts)
    return " ".join("".join(parts).split())


def extract_sections(epub_path: str) -> list:
    """Extract the documents of an EPUB book and their detected titles.

    Args:
        epub_path: Path to the ``.epub`` file.

    Returns:
        A list of :class:`Section` objects, numbered from 1.
    """
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


def load_sections(epub_path: str) -> list:
    """Read an EPUB book and return its sections.

    Args:
        epub_path: Path to the ``.epub`` file.

    Returns:
        The list of :class:`Section` objects.

    Raises:
        TTSError: If EPUB support is not installed or the book cannot be
            read.
    """
    if not EPUB_SUPPORT:
        raise TTSError(
            "'ebooklib' or 'beautifulsoup4' is not installed. "
            "Run 'pip install ebooklib beautifulsoup4'"
        )
    if not os.path.exists(epub_path):
        raise TTSError(f"EPUB file '{epub_path}' not found.")
    try:
        return extract_sections(epub_path)
    except Exception as exc:
        raise TTSError(f"Cannot read EPUB file: {exc}") from exc


def get_section_body(section: Section) -> str:
    """Return the text of a section without its leading heading.

    Args:
        section: The section to read.

    Returns:
        The body text, or an empty string if the section is empty or too
        short to be worth reading.
    """
    full_text = get_plain_text(section.soup)
    if len(full_text) < MIN_SECTION_CHARS:
        return ""
    if section.has_heading and full_text.startswith(section.title):
        return full_text[len(section.title):].strip()
    return full_text


def find_start_index(sections: list, section_start: str | None) -> int:
    """Resolve ``--section-start`` to a section index.

    Args:
        sections: All the sections of the book.
        section_start: A section number, a case-insensitive title substring,
            or None to start from the first section.

    Returns:
        The index of the first section to process.

    Raises:
        TTSError: If no section matches.
    """
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
        f"Section '{section_start}' not found. "
        "Use --section-list to see available sections."
    )


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
    """Read sections aloud, one after another.

    Args:
        sections: The sections to read.
        total: Total number of sections in the book (for progress text).
        lang_code: ISO language code.
        voice: Voice identifier, or None for the language default.
        display: Highlight words on screen while reading.
        speed: Speech speed multiplier for the body text.
        device: ``"cpu"`` or ``"cuda"``.
        lexicon: Pronunciation dictionary applied to the text.

    Returns:
        The number of sections that failed.

    Raises:
        PlayerNotFoundError: If ``ffplay`` is not installed.
        GpuMemoryError: If the GPU runs out of memory.
    """
    failed = 0
    for sec in sections:
        print(f"[{sec.index}/{total}] Reading section: {sec.title}")
        try:
            title = lexicon.apply(sec.title) if sec.has_heading else ""
            body = lexicon.apply(get_section_body(sec))
            if title or body:
                synthesize_and_play(
                    body,
                    lang_code=lang_code,
                    voice=voice,
                    display=display,
                    speed=speed,
                    device=device,
                    title=title,
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
    """Export sections to numbered audio files, skipping finished ones.

    Args:
        sections: The sections to export.
        total: Total number of sections in the book (for progress text).
        output_dir: Existing directory where the files are written.
        lang_code: ISO language code.
        voice: Voice identifier, or None for the language default.
        speed: Speech speed multiplier for the body text.
        device: ``"cpu"`` or ``"cuda"``.
        audio_format: ``"flac"`` or ``"mp3"``; it is also the file extension.
        mp3_quality: MP3 variable-bitrate quality (0 best, 9 smallest).
        lexicon: Pronunciation dictionary applied to the text. File names
            are always built from the original titles.

    Returns:
        The number of sections that failed.

    Raises:
        GpuMemoryError: If the GPU runs out of memory.
    """
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
        f"\n[EPUB] Finished: {done} exported, {skipped} skipped, "
        f"{failed} failed. Files are in '{output_dir}/'."
    )
    return failed


def process_epub_book(
    epub_path: str,
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
    """List, read aloud or export the sections of an EPUB book.

    Args:
        epub_path: Path to the ``.epub`` file.
        lang_code: ISO language code.
        voice: Voice identifier, or None for the language default.
        output_arg: None to read aloud; ``""`` to export into a directory
            named after the book; any other string is the output directory.
        section_list: If True, only list the sections and return.
        section_start: Section number or title substring to start from.
        display: Highlight words on screen while reading aloud.
        speed: Speech speed multiplier for the body text.
        device: ``"cpu"`` or ``"cuda"``.
        audio_format: ``"flac"`` or ``"mp3"`` (export mode only).
        mp3_quality: MP3 variable-bitrate quality, 0 (best) to 9 (smallest).
        lexicon: Pronunciation dictionary applied to the spoken text, or
            None to leave the text unchanged.

    Returns:
        The number of sections that failed (0 means complete success).

    Raises:
        TTSError: If the book cannot be read or the options are invalid.
    """
    if lexicon is None:
        lexicon = PronunciationDictionary({})
    sections = load_sections(epub_path)

    if section_list:
        print(f"\nSections found in '{epub_path}':")
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
            f"[EPUB Streaming Mode] Reading book starting from section "
            f"{start_idx}/{total}...\n"
        )
        failed = stream_sections(
            selected, total, lang_code, voice, display, speed, device,
            lexicon,
        )
        print("\n[EPUB] Streaming complete.")
        return failed

    if output_arg == "":
        output_dir = os.path.splitext(os.path.basename(epub_path))[0]
    else:
        output_dir = output_arg

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        raise TTSError(
            f"Cannot create directory '{output_dir}': {exc}"
        ) from exc
    print(f"[EPUB Export Mode] Output directory ready: '{output_dir}'")
    print(
        f"[EPUB Export Mode] Processing {len(selected)} sections starting "
        f"from {start_idx}...\n"
    )
    return export_sections(
        selected, total, output_dir, lang_code, voice, speed, device,
        audio_format, mp3_quality, lexicon,
    )
