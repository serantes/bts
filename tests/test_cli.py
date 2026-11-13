"""Command line parsing and dispatch."""

import io
import sys

import pytest

from bts import cli


def parse(*argv):
    return cli.build_parser().parse_args(list(argv))


def test_parser_defaults():
    args = parse()
    assert args.lang == "es" and args.speed == 1.0 and args.device == "auto"
    assert args.output is None and args.format is None
    assert args.mp3_quality == cli.DEFAULT_MP3_QUALITY
    assert args.max_pause == cli.DEFAULT_MAX_PAUSE_SECONDS


def test_parser_configures_completion_values():
    parser = cli.build_parser()
    actions = parser._option_string_actions
    assert actions["--voice"].completer is not None
    assert actions["--ipa"].completer is not None
    assert actions["--list-voices"].completer is not None
    assert actions["--test"].completer is not None


def test_parser_max_pause_accepts_zero_and_custom_value():
    assert parse("--max-pause", "0").max_pause == 0
    assert parse("--max-pause", "0.25").max_pause == 0.25


@pytest.mark.parametrize("value", ["-0.1", "nan", "inf", "abc"])
def test_parser_max_pause_rejects_invalid_values(value):
    with pytest.raises(SystemExit):
        parse("--max-pause", value)


def test_parser_output_without_value():
    assert parse("--output").output == ""
    assert parse("--output", "dir").output == "dir"


def test_parser_lang_case_insensitive_and_validated():
    assert parse("--lang", "US").lang == "us"
    with pytest.raises(SystemExit):
        parse("--lang", "xx")


def test_parser_epub_and_text_are_exclusive():
    with pytest.raises(SystemExit):
        parse("--epub", "a.epub", "--text", "a.txt")


def test_parser_rejects_legacy_txt_option():
    with pytest.raises(SystemExit):
        parse("--txt", "a.txt")


def test_parser_project_accepts_optional_name():
    assert parse("--epub", "book.epub", "--project").project == ""
    assert parse("--text", "book.txt", "--project", "book").project == "book"


def test_parser_tracks_explicit_build_overrides():
    defaults = parse()
    assert not defaults.lang_explicit
    assert not defaults.voice_explicit
    assert not defaults.format_explicit
    assert not defaults.dictionary_explicit

    args = parse(
        "--build", "book", "--lang", "fr", "--voice", "ff_siwis",
        "--format", "mp3", "--dictionary", "words",
    )
    assert args.lang_explicit
    assert args.voice_explicit
    assert args.format_explicit
    assert args.dictionary_explicit


def test_dictionary_display_enables_dictionary_review():
    args = parse("--dictionary", "words.dict", "--display")
    assert args.dictionary == "words.dict"
    assert args.dictionary_explicit


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "abc"])
def test_positive_float_rejects(value):
    with pytest.raises(SystemExit):
        parse("--speed", value)


def test_mp3_quality_range():
    assert parse("--mp3-quality", "9").mp3_quality == 9
    with pytest.raises(SystemExit):
        parse("--mp3-quality", "10")


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exit_info:
        parse("--version")
    assert exit_info.value.code == 0
    assert "BTS - Book To Speech" in capsys.readouterr().out


# --- --ipa parser tests -----------------------------------------------------


def test_parser_ipa_absent_gives_none():
    """Without --ipa the attribute is None."""
    assert parse().ipa is None


def test_parser_ipa_bare_gives_empty_string():
    """--ipa without LANG sets ipa to '' (const)."""
    assert parse("--ipa").ipa == ""


def test_parser_ipa_with_lang_stores_lowercased_code():
    """--ipa LANG stores the lowercase language code."""
    assert parse("--ipa", "US").ipa == "us"
    assert parse("--ipa", "en").ipa == "en"


def test_parser_ipa_unsupported_lang_errors(spy):
    """--ipa with an unknown language code exits with code 2."""
    with pytest.raises(SystemExit) as exc:
        run("--analyze", "--text", "a.txt", "--ipa", "xx")
    assert exc.value.code == 2


def test_run_analyze_with_ipa_lang_dispatches(monkeypatch):
    """--analyze --ipa LANG reaches run_analysis with ipa set to the code."""
    captured = {}

    def fake_analysis(args):
        captured["ipa"] = args.ipa
        captured["lang"] = args.lang
        return 0

    monkeypatch.setattr(cli, "run_analysis", fake_analysis)
    run("--analyze", "--text", "a.txt", "--lang", "es", "--ipa", "us")
    assert captured == {"ipa": "us", "lang": "es"}


# --- run -------------------------------------------------------------------


@pytest.fixture
def spy(monkeypatch):
    """Replace every heavy action with a recorder.

    Also stubs ``try_daemon_request`` to always return ``None`` so the tests
    are independent of whether a BTS daemon happens to be running in the
    background.  Without this stub the CLI would delegate to the daemon and
    bypass the monkeypatched synthesis functions, causing KeyError failures.
    """
    import bts.client as _client

    calls = {}
    monkeypatch.setattr(cli, "prepare_synthesis", lambda args, **kw: "cpu")
    monkeypatch.setattr(
        cli, "synthesize_and_play", lambda text, **kw: calls.update(
            play=(text, kw)
        )
    )
    monkeypatch.setattr(
        cli, "synthesize_text_to_file",
        lambda text, path, **kw: calls.update(file=(text, path, kw)),
    )
    monkeypatch.setattr(
        cli, "process_epub_book", lambda **kw: calls.update(epub=kw) or 0
    )
    # Prevent delegation to a live daemon so the stubs above are always
    # reached.
    monkeypatch.setattr(_client, "try_daemon_request", lambda req: None)
    return calls


def run(*argv):
    parser = cli.build_parser()
    return cli.run(parser.parse_args(list(argv)), parser)


def test_run_plain_text_plays(spy):
    assert run(
        "hola", "--speed", "1.2", "--lang", "us", "--max-pause", "0.2"
    ) == 0
    text, kw = spy["play"]
    assert text == "hola" and kw["speed"] == 1.2 and kw["lang_code"] == "us"
    assert kw["max_pause_seconds"] == 0.2


def test_run_text_applies_dictionary(spy, tmp_path):
    dictionary = tmp_path / "d.txt"
    dictionary.write_text("hola|adios\n", encoding="utf-8")
    run("hola mundo", "--dictionary", str(dictionary))
    assert spy["play"][0] == "adios mundo"


def test_dictionary_display_runs_interactive_dictionary_review(
    monkeypatch, tmp_path
):
    dictionary = tmp_path / "words.dict"
    dictionary.write_text("hola|adios\n", encoding="utf-8")
    captured = {}
    monkeypatch.setattr(cli, "prepare_synthesis", lambda *a, **kw: "cpu")

    def review(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("bts.review.review_dictionary", review)
    assert run("--dictionary", str(dictionary), "--display") == 0
    assert captured["path"] == str(dictionary)
    assert captured["device"] == "cpu"


def test_run_txt_to_file_defaults_to_txt_name(spy, tmp_path):
    source = tmp_path / "story.txt"
    source.write_text("hello", encoding="utf-8")
    run("--text", str(source), "--output")
    text, path, kw = spy["file"]
    assert text == "hello" and path == "story.flac"
    assert kw["audio_format"] == "flac"


def test_run_output_name_and_mp3(spy):
    run("hola", "--output", "x.mp3", "--mp3-quality", "2")
    _, path, kw = spy["file"]
    assert path == "x.mp3" and kw["audio_format"] == "mp3"
    assert kw["mp3_quality"] == 2


def test_run_output_without_name_for_plain_text(spy):
    with pytest.raises(cli.TTSError, match="needs a file name"):
        run("hola", "--output")


def test_run_reads_stdin_dash(spy, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("  from pipe \n"))
    run("-")
    assert spy["play"][0] == "from pipe"


def test_run_empty_stdin_is_error(spy, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("   "))
    with pytest.raises(cli.TTSError, match="empty input"):
        run("-")


def test_run_no_input_prints_help(spy, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_stdin_has_data", lambda timeout=2.0: False)
    assert run() == 1
    assert "usage:" in capsys.readouterr().out


def test_stdin_has_data_false_for_tty(monkeypatch):
    class Tty:
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdin", Tty())
    assert cli._stdin_has_data() is False


def test_stdin_has_data_with_real_pipe(monkeypatch):
    import os

    read_fd, write_fd = os.pipe()
    try:
        monkeypatch.setattr(sys, "stdin", os.fdopen(read_fd))
        assert cli._stdin_has_data(0.05) is False  # silent open pipe
        os.write(write_fd, b"x")
        assert cli._stdin_has_data(0.05) is True
    finally:
        os.close(write_fd)


def test_run_epub_passes_options(spy):
    run(
        "--epub", "b.epub", "--section-start", "3", "--format", "mp3",
        "--output", "out", "--speed", "0.9", "--max-pause", "0.25",
    )
    kw = spy["epub"]
    assert kw["epub_path"] == "b.epub" and kw["section_start"] == "3"
    assert kw["audio_format"] == "mp3" and kw["output_arg"] == "out"
    assert kw["speed"] == 0.9
    assert kw["max_pause_seconds"] == 0.25


@pytest.mark.parametrize(
    ("argv", "action"),
    [
        (("hola",), "play"),
        (("hola", "--output", "out.flac"), "save"),
        (("--epub", "b.epub"), "epub"),
    ],
)
def test_daemon_requests_include_max_pause(monkeypatch, argv, action):
    import bts.client

    requests = []
    monkeypatch.setattr(
        bts.client,
        "try_daemon_request",
        lambda request: requests.append(request) or 0,
    )

    assert run(*argv, "--max-pause", "0.25") == 0
    assert requests[0]["action"] == action
    assert requests[0]["max_pause"] == 0.25


def test_run_epub_failures_give_status_1(monkeypatch, spy):
    monkeypatch.setattr(cli, "process_epub_book", lambda **kw: 2)
    assert run("--epub", "b.epub") == 1


def test_run_project_creates_project_without_synthesis(monkeypatch, spy):
    calls = []
    monkeypatch.setattr(
        cli,
        "create_project",
        lambda source_file, project_name, source_type: (
            calls.append((source_file, project_name, source_type))
            or "book.bts"
        ),
    )

    assert run("--epub", "book", "--project") == 0
    assert calls == [("book.epub", None, "epub")]
    assert "epub" not in spy


def test_run_project_requires_file_source():
    with pytest.raises(SystemExit) as exit_info:
        run("--project")
    assert exit_info.value.code == 2


def test_run_build_passes_explicit_overrides(monkeypatch):
    captured = {}

    def fake_build(project_name, **kwargs):
        captured["project_name"] = project_name
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "build_project", fake_build)
    assert run(
        "--build", "book", "--output", "audio", "--lang", "fr",
        "--voice", "ff_siwis", "--format", "mp3",
        "--dictionary", "words",
    ) == 0
    assert captured["project_name"] == "book"
    assert captured["output_dir"] == "audio"
    assert captured["language_override"] == "fr"
    assert captured["voice_override"] == "ff_siwis"
    assert captured["format_override"] == "mp3"
    assert captured["dictionary_override"] == "words"


def test_run_sections_list_skips_device_preparation(monkeypatch, spy):
    def boom(*a, **k):
        raise AssertionError("not needed")

    monkeypatch.setattr(cli, "prepare_synthesis", boom)
    assert run("--epub", "b.epub", "--sections-list") == 0
    assert spy["epub"]["sections_list"] is True


@pytest.mark.parametrize(
    "argv",
    [
        ("--sections-list",),
        ("--section-start", "2"),
        ("--ipa",),
        ("--analyze",),
        ("hola", "--epub", "b.epub"),
    ],
)
def test_run_usage_errors(spy, argv):
    with pytest.raises(SystemExit) as exit_info:
        run(*argv)
    assert exit_info.value.code == 2


def test_run_listings(capsys):
    assert run("--list-languages") == 0
    assert run("--list-voices") == 0
    assert run("--list-voices", "es") == 0
    assert "ef_dora" in capsys.readouterr().out


def test_run_test_voices(monkeypatch, spy):
    monkeypatch.setattr(cli, "run_voice_tests", lambda lang, device: 1)
    assert run("--test", "es") == 1
    monkeypatch.setattr(cli, "run_voice_tests", lambda lang, device: 0)
    assert run("--test", "ES") == 0
    with pytest.raises(cli.TTSError):
        run("--test", "xx")


def test_run_analyze_dispatch(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_analysis", lambda args: 0)
    assert run("--analyze", "--text", "a.txt", "--output", "x") == 0
    assert "--output is ignored" in capsys.readouterr().err


# --- prepare_synthesis -------------------------------------------------------


def test_prepare_synthesis_checks_requirements(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "require_ffplay", lambda: calls.append("ffplay"))
    monkeypatch.setattr(cli, "require_mp3_encoder",
                        lambda: calls.append("mp3"))
    monkeypatch.setattr(cli, "resolve_device", lambda requested: requested)
    args = parse("--device", "auto")
    assert cli.prepare_synthesis(args, True, "es", needs_mp3=True) == "auto"
    assert calls == ["ffplay", "mp3"]
    calls.clear()
    cli.prepare_synthesis(args, False, "es")
    assert calls == []


def test_prepare_synthesis_validates_voice(monkeypatch):
    monkeypatch.setattr(cli, "resolve_device", lambda requested: "cpu")
    with pytest.raises(cli.TTSError, match="Unknown voice"):
        cli.prepare_synthesis(parse("--voice", "zzz"), False, "es")


def test_load_lexicon(tmp_path, capsys):
    assert len(cli.load_lexicon(parse())) == 0
    path = tmp_path / "d.txt"
    path.write_text("a|b\n", encoding="utf-8")
    lexicon = cli.load_lexicon(parse("--dictionary", str(path)))
    assert len(lexicon) == 1
    assert "Loaded 1 pronunciation entries" in capsys.readouterr().out


# --- main --------------------------------------------------------------------


def test_main_status_codes(monkeypatch, capsys):
    def raiser(error):
        def fail(args, parser):
            raise error

        return fail

    monkeypatch.setattr(cli, "run", raiser(cli.TTSError("bad")))
    assert cli.main(["x"]) == 1
    assert "Error: bad" in capsys.readouterr().err

    monkeypatch.setattr(cli, "run", raiser(cli.GpuMemoryError("full")))
    assert cli.main(["x"]) == cli.EXIT_GPU_BUSY == 0
    assert "Warning: full" in capsys.readouterr().err

    monkeypatch.setattr(cli, "run", raiser(KeyboardInterrupt()))
    assert cli.main(["x"]) == 0
    assert "Ctrl+C" in capsys.readouterr().out


def test_main_end_to_end_listing(capsys):
    assert cli.main(["--list-languages"]) == 0
    assert "Spanish" in capsys.readouterr().out
