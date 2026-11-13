"""Interactive pronunciation-dictionary review."""

import pytest

from bts import constants, review, synthesis
from bts.dictionary import PronunciationDictionary
from bts.numbers import prepare_text


def test_review_repeats_deletes_keeps_and_resumes_at_pending_line(
    monkeypatch, tmp_path
):
    path = tmp_path / "words.dict"
    path.write_text(
        "# User comment\nalpha|uno\nbeta|dos\ngamma|tres\n",
        encoding="utf-8",
    )
    spoken = []
    monkeypatch.setattr(
        review,
        "_speak_entry",
        lambda line, *args: spoken.append(line.split("|", 1)[0]),
    )
    answers = iter(["R", "Y", "N", "A", "Y"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    options = {
        "lang_code": "es",
        "voice": None,
        "speed": 1.0,
        "device": "cpu",
        "max_pause_seconds": 0.5,
    }
    assert review.review_dictionary(str(path), **options) == 0
    assert spoken == ["alpha", "alpha", "beta", "gamma"]
    assert path.read_text(encoding="utf-8") == (
        "# User comment\nbeta|dos\n# BTS REVIEW PENDING\ngamma|tres\n"
    )

    spoken.clear()
    answers = iter(["Y", "Y"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert review.review_dictionary(str(path), **options) == 0
    assert spoken == ["gamma"]
    assert path.read_text(encoding="utf-8") == "# User comment\nbeta|dos\n"


def test_review_discards_changes_when_user_declines_save(
    monkeypatch, tmp_path
):
    path = tmp_path / "words.dict"
    original = "alpha|uno\n"
    path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(review, "_speak_entry", lambda *args: None)
    answers = iter(["Y", "N"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    assert review.review_dictionary(
        str(path),
        lang_code="es",
        voice=None,
        speed=1.0,
        device="cpu",
        max_pause_seconds=0.5,
    ) == 0
    assert path.read_text(encoding="utf-8") == original


def test_abort_discards_changes_and_pending_marker_when_not_saved(
    monkeypatch, tmp_path
):
    path = tmp_path / "words.dict"
    original = "alpha|uno\nbeta|dos\n"
    path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(review, "_speak_entry", lambda *args: None)
    answers = iter(["Y", "A", "N"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    assert review.review_dictionary(
        str(path),
        lang_code="es",
        voice=None,
        speed=1.0,
        device="cpu",
        max_pause_seconds=0.5,
    ) == 0
    assert path.read_text(encoding="utf-8") == original


def test_speaking_review_entry_compares_source_and_replacement(
    monkeypatch, capsys
):
    spoken = {}
    monkeypatch.setattr(
        review,
        "synthesize_and_play",
        lambda text, **kwargs: spoken.update(text=text, **kwargs),
    )

    review._speak_entry(
        "alpha|uno\n",
        lang_code="es",
        voice=None,
        speed=1.0,
        device="cpu",
        max_pause_seconds=0.5,
    )

    assert capsys.readouterr().out == "alpha|uno\n"
    assert spoken["text"] == "uno"
    assert spoken["display"] is False
    assert spoken["title"] == "alpha"
    assert spoken["title_speed_factor"] == 1.0


def test_speaking_review_entry_compares_source_and_ipa(
    monkeypatch, capsys
):
    spoken = {}
    monkeypatch.setattr(
        review,
        "synthesize_and_play",
        lambda text, **kwargs: spoken.update(text=text, **kwargs),
    )

    review._speak_entry(
        "alpha|[ˈælfə]\n",
        lang_code="es",
        voice=None,
        speed=1.0,
        device="cpu",
        max_pause_seconds=0.5,
    )

    assert capsys.readouterr().out == "alpha|[ˈælfə]\n"
    assert spoken["title"] == "alpha"
    assert spoken["text"] == "[alpha](/ˈælfə/)"


@pytest.mark.parametrize(
    ("word", "replacement", "expected_source", "expected_ipa"),
    [
        ("asus", "[ˈAsəs]", "asus", "ˈAsəs"),
        ("asus", "[asus](/ˈAsəs/)", "asus", "ˈAsəs"),
        ("3", "[tɾes]", "tres", "tɾes"),
    ],
)
def test_review_source_matches_normal_synthesis(
    monkeypatch, fake_pipeline, fake_player,
    word, replacement, expected_source, expected_ipa,
):
    monkeypatch.setattr(synthesis, "get_pipeline", lambda lang, dev: fake_pipeline)
    monkeypatch.setattr(synthesis, "AudioPlayer", fake_player)
    speed = 1.3
    synthesis.synthesize_and_play(
        prepare_text(word, "es", PronunciationDictionary({})),
        lang_code="es",
        speed=speed,
    )
    normal_call = fake_pipeline.calls.pop()
    normal_audio = fake_player.instances[0].writes[0]

    review._speak_entry(
        f"{word}|{replacement}\n", "es", None, speed, "cpu", 0
    )

    assert normal_call == (expected_source, "ef_dora", speed)
    assert fake_pipeline.calls == [normal_call]
    assert fake_pipeline.token_calls == [(expected_ipa, "ef_dora", speed)]
    writes = fake_player.instances[1].writes
    assert len(writes) == 3
    assert (writes[0] == normal_audio).all()
    assert len(writes[1]) == int(
        constants.SAMPLE_RATE * constants.TITLE_PAUSE_SECONDS
    )
    assert not writes[1].any()
