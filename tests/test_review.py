"""Interactive pronunciation-dictionary review."""

from bts import review


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


def test_speaking_review_entry_shows_only_plain_line(monkeypatch, capsys):
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
