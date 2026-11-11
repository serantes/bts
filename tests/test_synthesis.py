"""Speech generation: phoneme injection, playback and file export."""

import pytest
import soundfile as sf

from bts import constants, errors, phonemes, synthesis, textio


@pytest.fixture
def use_pipeline(monkeypatch, fake_pipeline):
    monkeypatch.setattr(synthesis, "get_pipeline", lambda lang, dev: fake_pipeline)
    return fake_pipeline


# --- generate_speech ---------------------------------------------------------


def test_generate_speech_plain(fake_pipeline):
    out = list(synthesis.generate_speech(fake_pipeline, "e", "hola", "ef_dora", 1.0))
    assert [g for g, _ in out] == ["hola"]
    assert fake_pipeline.calls == [("hola", "ef_dora", 1.0)]


def test_generate_speech_english_passes_markup_through(fake_pipeline):
    text = "say [mr](/mˈɪ/) now"
    list(synthesis.generate_speech(fake_pipeline, "a", text, "af_heart", 1.0))
    assert fake_pipeline.calls[0][0] == text
    assert not fake_pipeline.token_calls


def test_generate_speech_other_language_injects_phonemes(fake_pipeline):
    text = "hola [mr](/mˈɪ/) adios"
    out = list(synthesis.generate_speech(fake_pipeline, "e", text, "ef_dora", 1.3))
    assert fake_pipeline.token_calls == [("HOLA mˈɪ ADIOS", "ef_dora", 1.3)]
    assert out[0][0] == "hola mr adios"
    assert not fake_pipeline.calls


class PlainPipeline:
    """A pipeline of an old kokoro: no ``generate_from_tokens``."""

    def __init__(self, inner):
        self.inner = inner
        self.g2p = inner.g2p

    def __call__(self, *args, **kwargs):
        return self.inner(*args, **kwargs)


def test_generate_speech_falls_back_without_generate_from_tokens(
    fake_pipeline, capsys
):
    pipeline = PlainPipeline(fake_pipeline)
    list(synthesis.generate_speech(pipeline, "e", "hola [mr](/x/)", "v", 1.0))
    assert fake_pipeline.calls[0][0] == "hola mr"
    assert "cannot synthesize phonemes" in capsys.readouterr().err


def test_chunk_phonemes_too_long_returns_none(fake_pipeline, capsys):
    chunk = "[a](/" + "x" * (constants.MAX_PHONEMES + 1) + "/)"
    assert phonemes.chunk_phonemes(fake_pipeline, chunk) is None
    assert "too long" in capsys.readouterr().err


def test_chunk_phonemes_g2p_failure_returns_none(fake_pipeline, capsys):
    fake_pipeline.g2p = lambda text: (_ for _ in ()).throw(ValueError("x"))
    assert phonemes.chunk_phonemes(fake_pipeline, "hi [a](/b/)") is None
    assert "could not phonemize" in capsys.readouterr().err


def test_join_phonemes_no_space_before_punctuation():
    assert phonemes.join_phonemes(["a", "", ",", "b", "."]) == "a, b."


def test_warn_once(capsys):
    errors.warn_once("same")
    errors.warn_once("same")
    assert capsys.readouterr().err.count("same") == 1


# --- synthesize_and_play -----------------------------------------------------


def test_play_text(monkeypatch, use_pipeline, fake_player):
    monkeypatch.setattr(synthesis, "AudioPlayer", fake_player)
    synthesis.synthesize_and_play("hola", lang_code="es")
    writes = fake_player.instances[0].writes
    assert len(writes) == 1 and len(writes[0]) == 240
    assert use_pipeline.calls[0][1] == "ef_dora"


def test_play_title_then_pause_then_body(
    monkeypatch, use_pipeline, fake_player
):
    monkeypatch.setattr(synthesis, "AudioPlayer", fake_player)
    synthesis.synthesize_and_play("body", lang_code="es", speed=2.0, title="Title")
    assert len(fake_player.instances) == 1  # one stream
    writes = fake_player.instances[0].writes
    assert len(writes) == 3
    pause = int(constants.SAMPLE_RATE * constants.TITLE_PAUSE_SECONDS)
    assert len(writes[1]) == pause and not writes[1].any()
    assert use_pipeline.calls[0] == ("Title", "ef_dora", 2.0 * constants.TITLE_SPEED)
    assert use_pipeline.calls[1] == ("body", "ef_dora", 2.0)


def test_play_title_only(monkeypatch, use_pipeline, fake_player):
    monkeypatch.setattr(synthesis, "AudioPlayer", fake_player)
    synthesis.synthesize_and_play("", lang_code="es", title="Only")
    assert len(fake_player.instances[0].writes) == 2


def test_play_unsupported_language():
    with pytest.raises(errors.TTSError, match="Unsupported"):
        synthesis.synthesize_and_play("x", lang_code="xx")


def test_play_wraps_generic_errors(monkeypatch, use_pipeline, fake_player):
    monkeypatch.setattr(synthesis, "AudioPlayer", fake_player)
    monkeypatch.setattr(synthesis, "is_cuda_oom", lambda exc: False)
    monkeypatch.setattr(
        synthesis, "generate_speech",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("model broke")),
    )
    with pytest.raises(errors.TTSError, match="model broke"):
        synthesis.synthesize_and_play("x", lang_code="es")


def test_play_translates_cuda_oom(monkeypatch, use_pipeline, fake_player):
    class Boom(fake_player):
        def write(self, samples):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(synthesis, "AudioPlayer", Boom)
    monkeypatch.setattr(synthesis, "is_cuda_oom", lambda exc: True)

    class T:
        class cuda:
            @staticmethod
            def empty_cache():
                pass

    monkeypatch.setattr(synthesis, "get_torch", lambda: T)
    with pytest.raises(errors.GpuMemoryError):
        synthesis.synthesize_and_play("x", lang_code="es")


def test_play_with_display_shows_text(
    monkeypatch, use_pipeline, fake_player, capsys
):
    monkeypatch.setattr(synthesis, "AudioPlayer", fake_player)
    synthesis.synthesize_and_play("hola mundo", lang_code="es", display=True)
    assert "-> hola mundo" in capsys.readouterr().out


# --- export ------------------------------------------------------------------


def test_export_section_flac(use_pipeline, tmp_path, capsys):
    path = str(tmp_path / "s.flac")
    synthesis.synthesize_epub_section_to_file(
        "Title", "One. Two.", path, lang_code="es"
    )
    data, rate = sf.read(path)
    assert rate == constants.SAMPLE_RATE and len(data) > 0
    assert not (tmp_path / "s.flac.tmp").exists()
    assert "100%" in capsys.readouterr().out


def test_export_section_no_audio_leaves_nothing(
    monkeypatch, use_pipeline, tmp_path
):
    path = str(tmp_path / "s.flac")
    with pytest.raises(errors.TTSError, match="No audio"):
        synthesis.synthesize_epub_section_to_file("", "   ", path, lang_code="es")
    assert list(tmp_path.iterdir()) == []


def test_export_section_failure_removes_temp(
    monkeypatch, use_pipeline, tmp_path
):
    monkeypatch.setattr(synthesis, "is_cuda_oom", lambda exc: False)
    monkeypatch.setattr(
        synthesis, "generate_speech",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad")),
    )
    path = str(tmp_path / "s.flac")
    with pytest.raises(errors.TTSError, match="bad"):
        synthesis.synthesize_epub_section_to_file("", "text", path)
    assert list(tmp_path.iterdir()) == []


def test_export_section_unsupported_language(tmp_path):
    with pytest.raises(errors.TTSError):
        synthesis.synthesize_epub_section_to_file(
            "", "x", str(tmp_path / "a.flac"), lang_code="xx"
        )


def test_synthesize_text_to_file_creates_directories(use_pipeline, tmp_path):
    path = str(tmp_path / "new" / "dir" / "out.flac")
    synthesis.synthesize_text_to_file("hola", path, lang_code="es")
    assert sf.info(path).frames > 0


def test_synthesize_text_to_file_replaces_existing(use_pipeline, tmp_path):
    path = tmp_path / "out.flac"
    path.write_bytes(b"old")
    synthesis.synthesize_text_to_file("hola", str(path), lang_code="es")
    assert sf.info(str(path)).frames > 0


# --- text files and output names ----------------------------------------------


def test_read_text_file(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("\ufeff  line1\nline2  \n", encoding="utf-8")
    assert textio.read_text_file(str(path)) == "line1\nline2"


def test_read_text_file_errors(tmp_path):
    with pytest.raises(errors.TTSError, match="not found"):
        textio.read_text_file(str(tmp_path / "missing.txt"))
    empty = tmp_path / "e.txt"
    empty.write_text("  \n")
    with pytest.raises(errors.TTSError, match="empty"):
        textio.read_text_file(str(empty))
    bad = tmp_path / "b.txt"
    bad.write_bytes(b"\xff\xfe\xfa")
    with pytest.raises(errors.TTSError, match="UTF-8"):
        textio.read_text_file(str(bad))


def test_resolve_audio_output(tmp_path):
    assert textio.resolve_audio_output("a.mp3", None, None) == ("a.mp3", "mp3")
    assert textio.resolve_audio_output("A.FLAC", "flac", None) == (
        "A.FLAC", "flac",
    )
    assert textio.resolve_audio_output("a", "mp3", None) == ("a.mp3", "mp3")
    assert textio.resolve_audio_output("a", None, None) == ("a.flac", "flac")
    assert textio.resolve_audio_output("", None, "story") == (
        "story.flac", "flac",
    )


def test_resolve_audio_output_errors(tmp_path):
    with pytest.raises(errors.TTSError, match="contradicts"):
        textio.resolve_audio_output("a.mp3", "flac", None)
    with pytest.raises(errors.TTSError, match="needs a file name"):
        textio.resolve_audio_output("", None, None)
    (tmp_path / "dir.flac").mkdir()
    with pytest.raises(errors.TTSError, match="directory"):
        textio.resolve_audio_output(str(tmp_path / "dir.flac"), None, None)
