"""Interactive pronunciation-dictionary review."""

import contextlib
import os
import select
import shutil
import subprocess
import sys
import threading
import time

import pytest

from bts import constants, review, synthesis
from bts.dictionary import PronunciationDictionary
from bts.numbers import prepare_text
from bts.settings import current_settings, use_settings


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


@pytest.mark.parametrize("save", ["Y", "N"])
def test_review_edits_then_repeats_and_saves_or_discards(
    monkeypatch, tmp_path, capsys, save
):
    path = tmp_path / "words.dict"
    original = b"\xef\xbb\xbfalpha|uno\r\n"
    path.write_bytes(original)
    spoken = []
    monkeypatch.setattr(
        review, "_speak_entry", lambda line, *args: spoken.append(line)
    )
    edits = iter(["alpha|dos", "alpha|dos"])
    monkeypatch.setattr(review, "_edit_entry", lambda content: next(edits))
    answers = iter(["E", "E", "R", "N", save])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    review.review_dictionary(
        str(path), lang_code="es", voice=None, speed=1,
        device="cpu", max_pause_seconds=0.5,
    )

    assert spoken == ["\ufeffalpha|uno\r\n", "\ufeffalpha|dos\r\n"]
    assert capsys.readouterr().out.count("alpha|dos\n") == 2
    expected = b"\xef\xbb\xbfalpha|dos\r\n" if save == "Y" else original
    assert path.read_bytes() == expected


def test_review_edited_entry_is_kept_when_aborting_and_saving(
    monkeypatch, tmp_path
):
    path = tmp_path / "words.dict"
    path.write_text("alpha|uno\n", encoding="utf-8")
    monkeypatch.setattr(review, "_speak_entry", lambda *args: None)
    monkeypatch.setattr(review, "_edit_entry", lambda content: "alpha|dos")
    answers = iter(["E", "A", "Y"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    review.review_dictionary(
        str(path), lang_code="es", voice=None, speed=1,
        device="cpu", max_pause_seconds=0.5,
    )
    assert path.read_text(encoding="utf-8") == (
        "# BTS REVIEW PENDING\nalpha|dos\n"
    )


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        (["x", "\x1a", "redo", "\r"], "alpha|unox"),
        (["x", "\x1a", "\x19", "\r"], "alpha|unox"),
        (["left", "\x7f", "X", "\r"], "alpha|uXo"),
        (["x", "\x1b"], "alpha|uno"),
        (["x", "\x1a", "y", "redo", "\r"], "alpha|unoy"),
    ],
)
def test_editor_undo_redo_and_cancel(monkeypatch, keys, expected):
    monkeypatch.setattr(review.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(review, "_terminal", contextlib.nullcontext)
    key_iter = iter(keys)
    monkeypatch.setattr(review, "_read_terminal_key", lambda: next(key_iter))
    assert review._edit_entry("alpha|uno") == expected


@pytest.mark.skipif(os.name == "nt", reason="POSIX terminal sequences")
def test_terminal_decodes_ctrl_shift_z(monkeypatch):
    master, slave = os.openpty()
    try:
        with os.fdopen(slave, "r", encoding="utf-8") as terminal:
            monkeypatch.setattr(review.sys, "stdin", terminal)
            with review._terminal():
                os.write(master, b"\x1b[122;6u\x1b[27;6;122~\x1b[122;5u")
                assert review._read_terminal_key() == "redo"
                assert review._read_terminal_key() == "redo"
                assert review._read_terminal_key() == "\x1a"
    finally:
        os.close(master)


@pytest.mark.skipif(os.name == "nt", reason="POSIX pseudo-terminal")
def test_interactive_review_interrupts_and_edits(tmp_path):
    path = tmp_path / "words.dict"
    path.write_text("alpha|uno\n", encoding="utf-8")
    master, slave = os.openpty()
    script = """
import sys
from bts import review
from bts.synthesis import SynthesisCancelled
review.get_pipeline = lambda *args: None
def speak(line, *args, cancel_event, show):
    if not cancel_event.wait(5):
        raise RuntimeError("speech was not interrupted")
    raise SynthesisCancelled()
review._speak_entry = speak
review.review_dictionary(
    sys.argv[1], lang_code="es", voice=None, speed=1,
    device="cpu", max_pause_seconds=0.5,
)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", script, str(path)],
        stdin=slave, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    os.close(slave)

    def wait_for(text):
        output = b""
        deadline = time.monotonic() + 10
        while text not in output and time.monotonic() < deadline:
            if select.select([proc.stdout], [], [], 0.1)[0]:
                part = os.read(proc.stdout.fileno(), 4096)
                if not part:
                    break
                output += part
        assert text in output, output

    try:
        wait_for(b"[E]dit")
        os.write(master, b"e")
        wait_for(b"\x1b[>4;2m")
        os.write(master, b"x\x1a\x1b[122;6u\r")
        wait_for(b"[E]dit")
        os.write(master, b"n")
        wait_for(b"Save changes?")
        os.write(master, b"y")
        _, error = proc.communicate(timeout=10)
        assert proc.returncode == 0, error.decode()
        assert path.read_text(encoding="utf-8") == "alpha|unox\n"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        os.close(master)


@pytest.mark.skipif(
    os.name == "nt" or shutil.which("ffplay") is None,
    reason="requires POSIX pseudo-terminal and ffplay",
)
def test_review_prompt_survives_real_playback(tmp_path):
    path = tmp_path / "words.dict"
    path.write_text("alpha|uno\n", encoding="utf-8")
    master, slave = os.openpty()
    script = """
import sys
import numpy as np
from bts import review, synthesis
class Pipeline:
    def __call__(self, text, **kwargs):
        yield text, None, np.zeros(2400, dtype=np.float32)
synthesis.get_pipeline = lambda *args: Pipeline()
review.get_pipeline = synthesis.get_pipeline
play = review.synthesize_and_play
def speak(*args, **kwargs):
    play(*args, **kwargs)
    print("\\\\a", end="", flush=True)
review.synthesize_and_play = speak
review.review_dictionary(
    sys.argv[1], lang_code="es", voice=None, speed=1,
    device="cpu", max_pause_seconds=0.5,
)
"""
    env = os.environ.copy()
    env["SDL_AUDIODRIVER"] = "dummy"
    proc = subprocess.Popen(
        [sys.executable, "-c", script, str(path)],
        stdin=slave, stdout=slave, stderr=slave, env=env,
    )
    os.close(slave)

    def wait_for(marker):
        output = b""
        deadline = time.monotonic() + 10
        while marker not in output and time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                output += os.read(master, 4096)
        assert marker in output, output
        return output

    try:
        output = wait_for(b"\\a")
        assert output == (
            b"[INFO] Loading speech model...\r\nalpha|uno\r\n"
            b"Delete? [Y]es, [N]o, [R]epeat, [E]dit, [A]bort: \\a"
        )
        os.write(master, b"n")
        wait_for(b"Save changes?")
        os.write(master, b"n")
        proc.wait(timeout=10)
        assert proc.returncode == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        os.close(master)


def test_review_key_interrupts_speech(monkeypatch):
    monkeypatch.setattr(review.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(review, "_terminal", contextlib.nullcontext)
    started = threading.Event()
    cancelled = threading.Event()

    def speak(line, *args, cancel_event, show):
        started.set()
        assert cancel_event.wait(2)
        cancelled.set()
        raise synthesis.SynthesisCancelled()

    monkeypatch.setattr(review, "_speak_entry", speak)
    monkeypatch.setattr(review, "_key_ready", lambda timeout: started.wait(2))
    monkeypatch.setattr(review, "_read_terminal_key", lambda: "e")
    with review._SpeechWorker() as worker:
        assert review._speak_and_decide(
            "alpha|uno\n", worker=worker
        ) == "edit"
    assert cancelled.is_set()


def test_review_surfaces_speech_errors(monkeypatch):
    monkeypatch.setattr(review.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(review, "_terminal", contextlib.nullcontext)
    monkeypatch.setattr(
        review, "_speak_entry",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            review.TTSError("speech failed")
        ),
    )
    monkeypatch.setattr(review, "_key_ready", lambda timeout: False)
    with pytest.raises(review.TTSError, match="speech failed"):
        with review._SpeechWorker() as worker:
            review._speak_and_decide("alpha|uno\n", worker=worker)


def test_review_does_not_wait_for_cancelled_synthesis_before_edit(monkeypatch):
    monkeypatch.setattr(review.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(review, "_terminal", contextlib.nullcontext)
    started = threading.Event()
    finish = threading.Event()

    def speak(line, *args, cancel_event, show):
        started.set()
        assert finish.wait(2)
        assert cancel_event.is_set()
        raise synthesis.SynthesisCancelled()

    monkeypatch.setattr(review, "_speak_entry", speak)
    monkeypatch.setattr(review, "_key_ready", lambda timeout: started.wait(2))
    monkeypatch.setattr(review, "_read_terminal_key", lambda: "e")
    with review._SpeechWorker() as worker:
        try:
            assert review._speak_and_decide(
                "alpha|uno\n", worker=worker
            ) == "edit"
        finally:
            finish.set()


def test_speech_worker_serializes_and_skips_cancelled_entries(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    last_spoken = threading.Event()
    spoken = []

    def speak(line, *args, cancel_event, show):
        spoken.append(line)
        if line == "first":
            started.set()
            assert release.wait(2)
        else:
            last_spoken.set()

    monkeypatch.setattr(review, "_speak_entry", speak)
    with review._SpeechWorker() as worker:
        try:
            first = worker.start("first", ())
            assert started.wait(2)
            first.set()
            skipped = worker.start("skipped", ())
            skipped.set()
            worker.start("last", ())
            release.set()
            assert last_spoken.wait(2)
        finally:
            release.set()
    assert spoken == ["first", "last"]


def test_review_prepares_model_before_first_entry_once(monkeypatch, tmp_path):
    path = tmp_path / "words.dict"
    path.write_text("# comment\nalpha|uno\nbeta|dos\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(review.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        review, "get_pipeline", lambda lang, device: calls.append(
            ("prepare", lang, device)
        )
    )
    monkeypatch.setattr(
        review, "_speak_and_decide",
        lambda line, *args, worker: calls.append(("speak", line)) or "no",
    )
    monkeypatch.setattr(review, "_confirm_save", lambda: False)

    review.review_dictionary(
        str(path), lang_code="es", voice=None, speed=1,
        device="cpu", max_pause_seconds=0.5,
    )

    assert calls == [
        ("prepare", "e", "cpu"), ("speak", "alpha|uno\n"),
        ("speak", "beta|dos\n"),
    ]


def test_speech_worker_uses_request_settings(monkeypatch):
    captured = []
    finished = threading.Event()

    def speak(line, *args, cancel_event, show):
        captured.append(current_settings().repo_id)
        finished.set()

    monkeypatch.setattr(review, "_speak_entry", speak)
    with use_settings({"repo_id": "custom/model"}):
        with review._SpeechWorker() as worker:
            worker.start("alpha|uno", ())
            assert finished.wait(2)
    assert captured == ["custom/model"]


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
    assert spoken["show_player_status"] is False
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
