"""EPUB reading, section selection, streaming and export."""

import pytest
from bs4 import BeautifulSoup
from ebooklib import epub as _ebooklib_epub

from bts import dictionary, epub, errors
from bts.settings import use_settings


def make_section(index, title, body="x" * 50, heading=True):
    html = f"<html><body><h1>{title}</h1><p>{body}</p></body></html>"
    if not heading:
        html = f"<html><body><p>{body}</p></body></html>"
        title = f"Section {index}"
    return epub.Section(
        index, title, heading, BeautifulSoup(html, "html.parser")
    )


def make_epub(path, chapters):
    book = _ebooklib_epub.EpubBook()
    book.set_identifier("id")
    book.set_title("Book")
    book.set_language("es")
    items = []
    for number, (title, body) in enumerate(chapters, 1):
        item = _ebooklib_epub.EpubHtml(
            title=title, file_name=f"c{number}.xhtml", lang="es"
        )
        heading = f"<h1>{title}</h1>" if title else ""
        item.content = f"<html><body>{heading}<p>{body}</p></body></html>"
        book.add_item(item)
        items.append(item)
    book.toc = tuple(items)
    book.add_item(_ebooklib_epub.EpubNcx())
    book.add_item(_ebooklib_epub.EpubNav())
    book.spine = ["nav", *items]
    _ebooklib_epub.write_epub(str(path), book)


# --- text extraction ---------------------------------------------------------


def text_of(html):
    return epub.get_plain_text(BeautifulSoup(html, "html.parser"))


def test_inline_tags_do_not_split_words():
    assert text_of("<p><a>A</a>ndrew <b>is</b> here</p>") == "Andrew is here"


def test_block_tags_separate_text():
    assert text_of("<p>one</p><p>two</p>one<br/>x") == "one two one x"


def test_whitespace_and_nbsp_normalized():
    assert text_of("<p>a \n\t b\u00a0c</p>") == "a b c"


def test_comments_and_scripts_ignored():
    html = "<p>a<!-- hidden --></p><script>var x</script>"
    assert "hidden" not in text_of(html)


def test_sup_is_separated():
    assert text_of("<p>word<sup>1</sup></p>") == "word 1"


def test_sanitize_filename():
    assert epub.sanitize_filename('Cap. 1: "El/fin"?') == "Cap._1_Elfin"
    assert epub.sanitize_filename("???") == "section"
    assert len(epub.sanitize_filename("a" * 100)) == 50


# --- sections -------------------------------------------------------------


def test_extract_sections_from_real_epub(tmp_path):
    path = tmp_path / "b.epub"
    make_epub(path, [("Uno", "Texto del primer capitulo."), ("", "Sin titulo")])
    sections = epub.load_sections(str(path))
    titles = [s.title for s in sections if s.has_heading]
    assert "Uno" in titles
    assert any(not s.has_heading for s in sections)
    assert [s.index for s in sections] == list(range(1, len(sections) + 1))


def test_load_sections_missing_file(tmp_path):
    with pytest.raises(errors.TTSError, match="not found"):
        epub.load_sections(str(tmp_path / "no.epub"))


def test_load_sections_corrupt_file(tmp_path):
    path = tmp_path / "bad.epub"
    path.write_text("not a zip")
    with pytest.raises(errors.TTSError, match="Cannot read EPUB"):
        epub.load_sections(str(path))


def test_load_sections_without_epub_support(monkeypatch, tmp_path):
    monkeypatch.setattr(epub, "EPUB_SUPPORT", False)
    with pytest.raises(errors.TTSError, match="not installed"):
        epub.load_sections(str(tmp_path / "x.epub"))


def test_get_section_body_strips_heading():
    section = make_section(1, "Title", "Some body text here")
    assert epub.get_section_body(section) == "Some body text here"


def test_configured_minimum_section_length():
    section = make_section(1, "", "short", heading=False)
    assert epub.get_section_body(section) == ""
    with use_settings({"min_section_chars": 4}):
        assert epub.get_section_body(section) == "short"


def test_get_section_body_without_heading():
    section = make_section(1, "", "Some body text here", heading=False)
    assert epub.get_section_body(section) == "Some body text here"


def test_get_section_body_too_short():
    assert epub.get_section_body(make_section(1, "T", "hi")) == ""


def test_find_start_index():
    sections = [make_section(1, "Intro"), make_section(2, "Capítulo Dos")]
    assert epub.find_start_index(sections, None) == 1
    assert epub.find_start_index(sections, "2") == 2
    assert epub.find_start_index(sections, "dos") == 2
    for bad in ("9", "zzz"):
        with pytest.raises(errors.TTSError, match="not found"):
            epub.find_start_index(sections, bad)


# --- streaming ---------------------------------------------------------------


def test_stream_sections_reads_title_and_body(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        epub, "synthesize_and_play", lambda body, **kw: calls.append((body, kw))
    )
    sections = [make_section(1, "Tit", "b" * 20), make_section(2, "", heading=False)]
    failed = epub.stream_sections(
        sections, 2, "es", None, False, 1.5, "cpu",
        dictionary.PronunciationDictionary({"Tit": "Titulo"}),
    )
    assert failed == 0
    assert calls[0][1]["title"] == "Titulo" and calls[0][1]["speed"] == 1.5
    assert calls[1][1]["title"] == ""


def test_stream_sections_counts_failures_and_continues(monkeypatch, capsys):
    def flaky(body, **kw):
        if kw["title"] == "A":
            raise errors.TTSError("bad")

    monkeypatch.setattr(epub, "synthesize_and_play", flaky)
    sections = [make_section(1, "A"), make_section(2, "B")]
    failed = epub.stream_sections(
        sections, 2, "es", None, False, 1.0, "cpu",
        dictionary.PronunciationDictionary({}),
    )
    assert failed == 1
    assert "section 1" in capsys.readouterr().err


@pytest.mark.parametrize("error", [errors.PlayerNotFoundError, errors.GpuMemoryError])
def test_stream_sections_fatal_errors_propagate(monkeypatch, error):
    def fatal(body, **kw):
        raise error("x")

    monkeypatch.setattr(epub, "synthesize_and_play", fatal)
    with pytest.raises(error):
        epub.stream_sections(
            [make_section(1, "A")], 1, "es", None, False, 1.0, "cpu",
            dictionary.PronunciationDictionary({}),
        )


# --- export ---------------------------------------------------------------


def export(tmp_path, sections, **kw):
    return epub.export_sections(
        sections, len(sections), str(tmp_path), "es", None, 1.0, "cpu",
        kw.pop("audio_format", "flac"), 4,
        kw.pop("lexicon", dictionary.PronunciationDictionary({})),
    )


def test_export_sections_writes_names_and_skips(monkeypatch, tmp_path, capsys):
    written = []

    def fake(title, body, path, **kw):
        written.append((title, body, path, kw["audio_format"]))
        open(path, "wb").close()

    monkeypatch.setattr(epub, "synthesize_epub_section_to_file", fake)
    sections = [
        make_section(1, "Uno"),
        make_section(2, "Dos", body="hi"),  # too short
        make_section(3, "Tres"),
    ]
    assert export(tmp_path, sections, audio_format="mp3") == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "01_Uno.mp3", "03_Tres.mp3",
    ]
    # Second run: finished sections are skipped.
    written.clear()
    assert export(tmp_path, sections, audio_format="mp3") == 0
    assert written == []
    assert "Skipping (already completed)" in capsys.readouterr().out


def test_export_sections_applies_dictionary_but_not_to_filenames(
    monkeypatch, tmp_path
):
    seen = []
    monkeypatch.setattr(
        epub, "synthesize_epub_section_to_file",
        lambda title, body, path, **kw: seen.append((title, path)),
    )
    export(
        tmp_path, [make_section(1, "Mr")],
        lexicon=dictionary.PronunciationDictionary({"Mr": "Mister"}),
    )
    assert seen[0][0] == "Mister" and seen[0][1].endswith("01_Mr.flac")


def test_export_sections_failure_is_counted(monkeypatch, tmp_path):
    def fail(*a, **k):
        raise errors.TTSError("x")

    monkeypatch.setattr(epub, "synthesize_epub_section_to_file", fail)
    assert export(tmp_path, [make_section(1, "A")]) == 1


def test_export_sections_gpu_error_aborts(monkeypatch, tmp_path):
    def fail(*a, **k):
        raise errors.GpuMemoryError("x")

    monkeypatch.setattr(epub, "synthesize_epub_section_to_file", fail)
    with pytest.raises(errors.GpuMemoryError):
        export(tmp_path, [make_section(1, "A")])


# --- process_epub_book -----------------------------------------------------


@pytest.fixture
def three_sections(monkeypatch):
    sections = [make_section(i, f"T{i}") for i in (1, 2, 3)]
    monkeypatch.setattr(epub, "load_sections", lambda path: sections)
    return sections


def test_process_list_sections(three_sections, capsys):
    assert epub.process_epub_book("b.epub", "es", None, None, True, None, False) == 0
    out = capsys.readouterr().out
    assert "[1] T1" in out and "[3] T3" in out


def test_process_streaming_from_section(three_sections, monkeypatch):
    got = {}

    def fake(selected, total, *args):
        got["indexes"] = [s.index for s in selected]
        got["total"] = total
        return 0

    monkeypatch.setattr(epub, "stream_sections", fake)
    epub.process_epub_book("b.epub", "es", None, None, False, "2", False)
    assert got == {"indexes": [2, 3], "total": 3}


def test_process_export_default_dir_named_after_book(
    three_sections, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(epub, "export_sections", lambda *a: 0)
    epub.process_epub_book("/x/My Book.epub", "es", None, "", False, None, False)
    assert (tmp_path / "My Book").is_dir()


def test_process_export_unwritable_dir(three_sections, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(errors.TTSError, match="Cannot create directory"):
        epub.process_epub_book(
            "b.epub", "es", None, str(blocker / "sub"), False, None, False
        )
