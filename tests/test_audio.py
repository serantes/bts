"""Audio helpers, player, MP3/FLAC writers and the karaoke display."""

import struct
import subprocess

import numpy as np
import pytest
import soundfile as sf

from bts import audio, constants, errors, textio


def test_to_float32_from_list_and_tensorlike():
    assert audio.to_float32([0.5, 1]).dtype == np.float32

    class Tensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.array([0.25], dtype=np.float64)

    out = audio.to_float32(Tensor())
    assert out.dtype == np.float32 and out[0] == 0.25


def test_to_pcm16_clips_and_scales():
    data = audio.to_pcm16(np.array([0.0, 1.0, -1.0, 2.0, -2.0]))
    assert struct.unpack("<5h", data) == (0, 32767, -32767, 32767, -32767)


def test_streaming_wav_header():
    header = audio.streaming_wav_header()
    assert len(header) == 44
    assert header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    channels, rate = struct.unpack("<HI", header[22:28])
    assert (channels, rate) == (1, constants.SAMPLE_RATE)
    assert header[40:44] == b"\xff\xff\xff\xff"


def test_split_into_chunks_respects_lines_and_size():
    assert textio.split_into_chunks("Uno. Dos. Tres.\nCuatro.", 9) == [
        "Uno. Dos.", "Tres.", "Cuatro.",
    ]


def test_split_into_chunks_long_sentence_and_empty():
    long = "a" * 50
    assert textio.split_into_chunks(long, 10) == [long]
    assert textio.split_into_chunks("  \n\n ") == []


def test_split_into_chunks_cjk_terminators():
    assert textio.split_into_chunks("你好。再见！", 3) == ["你好。", "再见！"]


def test_wrap_words():
    assert audio.wrap_words(["aa", "bb", "cc"], 5) == [[0, 1], [2]]
    assert audio.wrap_words(["long-word"], 3) == [[0]]
    assert audio.wrap_words([], 10) == [[]]


# --- ffplay ---------------------------------------------------------------


class FakeStdin:
    def __init__(self, fail=False):
        self.data = b""
        self.closed = False
        self.fail = fail

    def write(self, payload):
        if self.fail:
            raise BrokenPipeError()
        self.data += payload

    def flush(self):
        pass

    def close(self):
        self.closed = True


class FakeProc:
    created = []

    def __init__(self, command, stdin=None, stderr=None, fail=False,
                 returncode=0, error=b""):
        self.command = command
        self.stdin = FakeStdin(fail)
        self.returncode = returncode
        self.error = error
        self.waited = self.terminated = self.killed = False
        FakeProc.created.append(self)

    def wait(self, timeout=None):
        self.waited = True
        return self.returncode

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def communicate(self):
        return b"", self.error


@pytest.fixture
def fake_popen(monkeypatch):
    FakeProc.created = []
    monkeypatch.setattr(subprocess, "Popen", FakeProc)
    return FakeProc


def test_require_ffplay(monkeypatch):
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    with pytest.raises(errors.PlayerNotFoundError):
        audio.require_ffplay()
    monkeypatch.setattr(audio.shutil, "which", lambda name: "/bin/x")
    audio.require_ffplay()


def test_player_is_lazy_and_sends_header_once(fake_popen):
    with audio.AudioPlayer() as player:
        assert not fake_popen.created
        player.write(np.zeros(2, dtype=np.float32))
        player.write(np.zeros(2, dtype=np.float32))
    proc = fake_popen.created[0]
    assert proc.command[0] == "ffplay"
    assert proc.stdin.data.count(b"RIFF") == 1
    assert len(proc.stdin.data) == 44 + 4 + 4
    assert proc.stdin.closed and proc.waited and not proc.terminated


def test_player_without_audio_launches_nothing(fake_popen):
    with audio.AudioPlayer():
        pass
    assert not fake_popen.created


def test_player_stops_on_exception(fake_popen):
    with pytest.raises(KeyError):
        with audio.AudioPlayer() as player:
            player.write(np.zeros(1, dtype=np.float32))
            raise KeyError("x")
    proc = fake_popen.created[0]
    assert proc.terminated and proc.stdin.closed


def test_player_broken_pipe_is_tts_error(monkeypatch):
    monkeypatch.setattr(
        subprocess, "Popen",
        lambda *a, **k: FakeProc(*a, fail=True, returncode=1, **k),
    )
    with pytest.raises(errors.TTSError, match="stopped unexpectedly"):
        with audio.AudioPlayer() as player:
            player.write(np.zeros(1, dtype=np.float32))


def test_player_missing_binary(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(subprocess, "Popen", missing)
    with pytest.raises(errors.PlayerNotFoundError):
        with audio.AudioPlayer() as player:
            player.write(np.zeros(1, dtype=np.float32))


# --- MP3 -------------------------------------------------------------------


def test_require_mp3_encoder(monkeypatch):
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    with pytest.raises(errors.TTSError, match="ffmpeg"):
        audio.require_mp3_encoder()

    monkeypatch.setattr(audio.shutil, "which", lambda name: "/bin/ffmpeg")
    for output, ok in (("... libmp3lame ...", True), ("aac only", False)):
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, _o=output, **k: subprocess.CompletedProcess(
                a, 0, stdout=_o, stderr=""
            ),
        )
        if ok:
            audio.require_mp3_encoder()
        else:
            with pytest.raises(errors.TTSError, match="libmp3lame"):
                audio.require_mp3_encoder()


def test_mp3_writer_pipes_pcm_to_ffmpeg(fake_popen, tmp_path):
    path = str(tmp_path / "a.mp3")
    with audio.Mp3Writer(path, quality=2) as writer:
        writer.write(np.zeros(3, dtype=np.float32))
    proc = fake_popen.created[0]
    assert proc.command[0] == "ffmpeg"
    assert "libmp3lame" in proc.command and "2" in proc.command
    assert proc.command[-1] == path
    assert len(proc.stdin.data) == 6


def test_mp3_writer_reports_ffmpeg_failure(monkeypatch):
    monkeypatch.setattr(
        subprocess, "Popen",
        lambda *a, **k: FakeProc(*a, returncode=1, error=b"bad", **k),
    )
    with pytest.raises(errors.TTSError, match="bad"):
        with audio.Mp3Writer("x.mp3"):
            pass


def test_mp3_writer_aborts_on_exception(fake_popen):
    with pytest.raises(ValueError):
        with audio.Mp3Writer("x.mp3"):
            raise ValueError()
    assert fake_popen.created[0].killed


def test_mp3_writer_missing_ffmpeg(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(subprocess, "Popen", missing)
    with pytest.raises(errors.TTSError, match="not installed"):
        with audio.Mp3Writer("x.mp3"):
            pass


def test_mp3_writer_write_failure(monkeypatch):
    monkeypatch.setattr(
        subprocess, "Popen",
        lambda *a, **k: FakeProc(*a, fail=True, error=b"died", **k),
    )
    with pytest.raises(errors.TTSError, match="died"):
        with audio.Mp3Writer("x.mp3") as writer:
            writer.write(np.zeros(1, dtype=np.float32))


# --- FLAC ------------------------------------------------------------------


def test_open_audio_writer_flac_roundtrip(tmp_path):
    path = str(tmp_path / "a.flac")
    samples = np.linspace(-0.5, 0.5, 2400, dtype=np.float32)
    with audio.open_audio_writer(path, "flac", 4) as out:
        out.write(samples)
    data, rate = sf.read(path)
    assert rate == constants.SAMPLE_RATE and len(data) == 2400


def test_open_audio_writer_mp3_type():
    assert isinstance(audio.open_audio_writer("a.mp3", "mp3", 3), audio.Mp3Writer)


# --- karaoke ---------------------------------------------------------------


def test_karaoke_non_interactive_prints(capsys):
    display = audio.KaraokeDisplay()
    display.submit("hola mundo", 1.0)
    display.submit("", 1.0)
    display.close()
    out = capsys.readouterr().out
    assert "-> hola mundo" in out


def test_karaoke_abort_skips_pending(capsys):
    display = audio.KaraokeDisplay()
    display._abort.set()
    display.submit("never shown", 1.0)
    display.close(abort=True)
    assert "never shown" not in capsys.readouterr().out


def test_karaoke_interactive_draws_highlight(monkeypatch, capsys):
    monkeypatch.setattr(audio.time, "sleep", lambda s: None)
    display = audio.KaraokeDisplay()
    display._interactive = True
    display.submit("uno dos", 0.1)
    display.close()
    out = capsys.readouterr().out
    assert audio.KaraokeDisplay.HIGHLIGHT in out
    assert "\033[1F" in out
