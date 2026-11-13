from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from bts import project
from bts.errors import TTSError


def read_project(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))["BTSProject"]


def test_create_project_from_epub(tmp_path, monkeypatch):
    source = tmp_path / "My Book.epub"
    monkeypatch.setattr(
        project,
        "load_sections",
        lambda _path: [
            SimpleNamespace(index=1, title="Chapter One"),
            SimpleNamespace(index=2, title="Chapter Two"),
        ],
    )

    project_path = project.create_project(str(source))
    output = tmp_path / "My Book.bts"
    data = read_project(output)
    contents = output.read_text(encoding="utf-8")

    assert project_path == str(tmp_path / "My Book.bts")
    assert contents.startswith("# BTS project file.\n")
    assert "# - A custom element is not in the source" in contents
    assert "  Language: default\n  Voice: default\n" in contents
    assert "  Elements:\n    - type: source\n" in contents
    assert list(data) == [
        "Title",
        "SourceFile",
        "Dictionary",
        "Language",
        "Voice",
        "OutputFormat",
        "FilenamePattern",
        "Elements",
    ]
    assert data == {
        "Title": "My Book",
        "SourceFile": str(source),
        "Dictionary": str(tmp_path / "My Book.dict"),
        "Language": "default",
        "Voice": "default",
        "OutputFormat": "default",
        "FilenamePattern": "{index:02d} - {name}",
        "Elements": [
            {"type": "source", "id": 1, "name": "Chapter One"},
            {"type": "source", "id": 2, "name": "Chapter Two"},
        ],
    }


def test_create_project_from_text_has_one_source_element(tmp_path):
    source = tmp_path / "story.txt"
    source.write_text("A valid source text.", encoding="utf-8")

    project_path = project.create_project(
        str(source), str(tmp_path / "reading"), source_type="text"
    )
    data = read_project(tmp_path / "reading.bts")

    assert project_path == str(tmp_path / "reading.bts")
    assert data["SourceFile"] == str(source)
    assert data["Elements"] == [
        {"type": "source", "id": 1, "name": "story"}
    ]


def test_create_project_keeps_explicit_extension(tmp_path):
    source = tmp_path / "story.txt"
    source.write_text("A valid source text.", encoding="utf-8")
    project_path = tmp_path / "reading.yaml"

    result = project.create_project(
        str(source), str(project_path), source_type="text"
    )

    assert result == str(project_path)
    assert project_path.is_file()


def test_create_project_does_not_overwrite_existing_file(tmp_path):
    source = tmp_path / "story.txt"
    source.write_text("A valid source text.", encoding="utf-8")
    project_path = tmp_path / "existing.bts"
    project_path.write_text("keep this", encoding="utf-8")

    with pytest.raises(TTSError, match="already exists"):
        project.create_project(
            str(source), str(project_path), source_type="text"
        )

    assert project_path.read_text(encoding="utf-8") == "keep this"


def write_build_project(tmp_path, source_file, elements, **overrides):
    settings = {
        "Title": "A Book",
        "SourceFile": str(source_file),
        "Dictionary": str(tmp_path / "book.dict"),
        "Language": "default",
        "Voice": "default",
        "OutputFormat": "default",
        "FilenamePattern": "{index:02d} - {name}",
        "Elements": elements,
    }
    settings.update(overrides)
    path = tmp_path / "build.bts"
    path.write_text(
        yaml.safe_dump({"BTSProject": settings}, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_build_project_ignores_missing_source_section_before_dictionary(
    tmp_path, monkeypatch, capsys
):
    source = tmp_path / "book.epub"
    project_file = write_build_project(
        tmp_path,
        source,
        [{"type": "source", "id": 99, "name": "Missing"}],
    )
    monkeypatch.setattr(
        project,
        "load_sections",
        lambda _path: [
            SimpleNamespace(index=1, title="Existing", has_heading=True)
        ],
    )

    assert project.build_project(str(project_file.with_suffix(""))) == 0
    captured = capsys.readouterr()
    assert "No buildable elements" in captured.out
    assert "does not exist" not in captured.err


def test_build_project_uses_defaults_and_warns_for_missing_dictionary(
    tmp_path, monkeypatch, capsys
):
    source_file = tmp_path / "story.txt"
    source_file.write_text("Once upon a time.", encoding="utf-8")
    project_file = write_build_project(
        tmp_path,
        source_file,
        [{"type": "source", "id": 1, "name": "default"}],
        Dictionary="default",
    )
    generated = []

    def fake_synthesize(title, body, output_path, **kwargs):
        generated.append((title, body, output_path, kwargs))
        Path(output_path).write_text("audio", encoding="utf-8")

    monkeypatch.setattr(project, "validate_voice", lambda *_args: None)
    monkeypatch.setattr(project, "resolve_device", lambda device: device)
    monkeypatch.setattr(
        project, "synthesize_epub_section_to_file", fake_synthesize
    )

    assert project.build_project(
        str(project_file), output_dir=str(tmp_path / "audio")
    ) == 0

    captured = capsys.readouterr()
    assert str(source_file.with_suffix(".dict")) in captured.err
    assert "does not exist" in captured.err
    assert len(generated) == 1
    title, body, output_path, kwargs = generated[0]
    assert title == ""
    assert body == "Once upon a time."
    assert Path(output_path).name == "01 - story.flac"
    assert kwargs["lang_code"] == "es"
    assert kwargs["voice"] is None
    assert kwargs["audio_format"] == "flac"


def test_build_project_config_defaults_yield_to_project_settings(
    tmp_path, monkeypatch
):
    source = tmp_path / "story.txt"
    source.write_text("Hello.", encoding="utf-8")
    project_file = write_build_project(
        tmp_path, source, [{"type": "source", "id": 1, "name": "default"}],
        Language="us", Dictionary="default",
    )
    generated = []
    monkeypatch.setattr(project, "validate_voice", lambda *_args: None)
    monkeypatch.setattr(project, "resolve_device", lambda device: device)
    monkeypatch.setattr(
        project, "synthesize_epub_section_to_file",
        lambda *args, **kwargs: generated.append((args, kwargs)),
    )

    assert project.build_project(
        str(project_file), output_dir=str(tmp_path / "audio"),
        default_language="fr", default_voice="ff_siwis", default_format="mp3",
    ) == 0
    args, kwargs = generated[0]
    assert kwargs["lang_code"] == "us"
    assert kwargs["voice"] == "ff_siwis"
    assert kwargs["audio_format"] == "mp3"
    assert str(args[2]).endswith(".mp3")


def test_build_project_uses_pattern_and_cli_overrides(
    tmp_path, monkeypatch, capsys
):
    source = tmp_path / "book.epub"
    project_dictionary = tmp_path / "book.dict"
    override_dictionary = tmp_path / "override.dict"
    project_dictionary.write_text("a|b\n", encoding="utf-8")
    override_dictionary.write_text("c|d\n", encoding="utf-8")
    project_file = write_build_project(
        tmp_path,
        source,
        [
            {
                "type": "source",
                "id": 1,
                "name": "default",
                "speech_before": "Previously",
                "speech_after": "Continuing",
            },
            {
                "type": "custom",
                "name": "Outro",
                "text": "Thank you.",
                "speech_before": "In conclusion",
            },
            {"type": "source", "id": 99, "name": "Missing"},
            {"type": "custom", "name": "", "text": "Skipped"},
        ],
        Language="en",
        Voice="bf_emma",
        OutputFormat="flac",
    )
    monkeypatch.setattr(
        project,
        "load_sections",
        lambda _path: [
            SimpleNamespace(index=1, title="Chapter One", has_heading=True)
        ],
    )
    monkeypatch.setattr(
        project, "get_section_body", lambda _section: "Chapter text."
    )
    monkeypatch.setattr(project, "validate_voice", lambda *_args: None)
    monkeypatch.setattr(project, "resolve_device", lambda _device: "cpu")
    monkeypatch.setattr(project, "require_mp3_encoder", lambda: None)
    monkeypatch.setattr(
        project,
        "load_dictionary",
        lambda *_args: project.PronunciationDictionary({}),
    )
    generated = []

    def fake_synthesize(title, body, path, **kwargs):
        generated.append((title, body, path, kwargs))
        Path(path).write_text("audio", encoding="utf-8")

    monkeypatch.setattr(
        project, "synthesize_epub_section_to_file", fake_synthesize
    )
    output_dir = tmp_path / "audio"

    assert project.build_project(
        str(project_file),
        output_dir=str(output_dir),
        language_override="es",
        voice_override="ef_dora",
        format_override="mp3",
        dictionary_override=str(override_dictionary),
    ) == 0

    assert [Path(item[2]).name for item in generated] == [
        "01 - Chapter One.mp3",
        "02 - Outro.mp3",
    ]
    assert generated[0][0:2] == (
        "Previously\nChapter One",
        "Chapter text.\nContinuing",
    )
    assert generated[1][0:2] == ("In conclusion", "Thank you.")
    assert all(item[3]["lang_code"] == "es" for item in generated)
    assert all(item[3]["voice"] == "ef_dora" for item in generated)
    err = capsys.readouterr().err
    assert err.count("overrides the project value") == 4
    assert "empty required field 'name'" in err
