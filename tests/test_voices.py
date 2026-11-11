"""Languages, voices and their listings."""

import pytest

from bts import errors, languages, synthesis


def test_default_voice_per_language():
    assert languages.get_default_voice("es") == "ef_dora"
    assert languages.get_default_voice("US") == "af_heart"
    assert languages.get_default_voice("uk") == "bf_emma"
    assert languages.get_default_voice("xx") == "ef_dora"


def test_every_language_has_voices_and_test_text():
    for code, info in languages.SUPPORTED_LANGUAGES.items():
        assert languages.ISO_TO_VOICE_KEY[code] in languages.ALL_VOICES
        assert code in languages.TEST_TEXTS
        voice = languages.get_default_voice(code)
        assert voice in languages.KNOWN_VOICES
        assert voice[0] == info["internal"]


def test_validate_voice_accepts_none_known_and_file(tmp_path):
    languages.validate_voice("es", None)
    languages.validate_voice("es", "ef_dora")
    custom = tmp_path / "mine.pt"
    custom.write_bytes(b"x")
    languages.validate_voice("es", str(custom))


def test_validate_voice_unknown():
    with pytest.raises(errors.TTSError, match="Unknown voice"):
        languages.validate_voice("es", "nope")


def test_validate_voice_other_language_warns(capsys):
    languages.validate_voice("es", "af_heart")
    assert "different language" in capsys.readouterr().err


def test_show_languages(capsys):
    languages.show_languages()
    out = capsys.readouterr().out
    assert "[es]" in out and "[zh]" in out


def test_show_voices_all_and_filtered(capsys):
    languages.show_voices("all")
    assert "am_adam" in capsys.readouterr().out
    languages.show_voices("fr")
    out = capsys.readouterr().out
    assert "ff_siwis" in out and "am_adam" not in out


def test_show_voices_unknown_language():
    with pytest.raises(errors.TTSError):
        languages.show_voices("xx")


def test_run_voice_tests_plays_every_voice(monkeypatch, capsys):
    played = []
    monkeypatch.setattr(
        synthesis, "synthesize_and_play",
        lambda text, lang_code, voice, device: played.append(voice),
    )
    assert synthesis.run_voice_tests("es") == 0
    assert played == languages.ALL_VOICES["es (Spanish)"]


def test_run_voice_tests_counts_failures(monkeypatch, capsys):
    def fail(text, lang_code, voice, device):
        raise errors.TTSError("boom")

    monkeypatch.setattr(synthesis, "synthesize_and_play", fail)
    assert synthesis.run_voice_tests("fr") == 1
    assert "boom" in capsys.readouterr().err


def test_run_voice_tests_propagates_fatal(monkeypatch):
    def fatal(text, lang_code, voice, device):
        raise errors.PlayerNotFoundError("no ffplay")

    monkeypatch.setattr(synthesis, "synthesize_and_play", fatal)
    with pytest.raises(errors.PlayerNotFoundError):
        synthesis.run_voice_tests("es")


def test_run_voice_tests_unsupported():
    with pytest.raises(errors.TTSError):
        synthesis.run_voice_tests("xx")
