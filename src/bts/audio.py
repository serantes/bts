"""Audio conversion, playback, MP3 writing and karaoke display."""

from __future__ import annotations

import contextlib
import queue
import shutil
import struct
import subprocess
import sys
import threading
import time

import numpy as np
import soundfile as sf

from bts.constants import DEFAULT_MP3_QUALITY, SAMPLE_RATE
from bts.errors import PlayerNotFoundError, TTSError


def to_float32(audio) -> np.ndarray:
    """Convert a Kokoro audio chunk (tensor or array) to a float32 array."""
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32)


def to_pcm16(samples: np.ndarray) -> bytes:
    """Convert float samples in [-1, 1] to little-endian 16-bit PCM."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def streaming_wav_header(
    sample_rate: int = SAMPLE_RATE, channels: int = 1, bits: int = 16
) -> bytes:
    """Build a WAV header of unknown length, for endless streaming.

    A WAV header declares the size of its audio data and players stop
    reading at that size, so sending one complete WAV file per audio chunk
    plays only the first chunk. Declaring the maximum size once and then
    sending raw PCM lets the player read until the pipe is closed.

    Args:
        sample_rate: Sampling rate in Hz.
        channels: Number of audio channels.
        bits: Bits per sample.

    Returns:
        The 44-byte header.
    """
    block_align = channels * bits // 8
    byte_rate = sample_rate * block_align
    unknown_size = 0xFFFFFFFF
    return (
        b"RIFF"
        + struct.pack("<I", unknown_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack(
            "<IHHIIHH", 16, 1, channels, sample_rate, byte_rate,
            block_align, bits,
        )
        + b"data"
        + struct.pack("<I", unknown_size)
    )


def require_ffplay() -> None:
    """Raise PlayerNotFoundError if ``ffplay`` is not on the PATH."""
    if shutil.which("ffplay") is None:
        raise PlayerNotFoundError(
            "'ffplay' (from ffmpeg) is not installed."
        )


class AudioPlayer:
    """Feed audio to ``ffplay`` as a single streaming WAV stream.

    ``ffplay`` is started lazily on the first :meth:`write`, so nothing is
    launched if synthesis fails before producing audio. Use it as a context
    manager: a normal exit waits until playback finishes, while an exception
    (including Ctrl+C) stops the player immediately.
    """

    COMMAND = [
        "ffplay",
        "-nodisp",
        "-autoexit",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
    ]

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> AudioPlayer:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        proc = self._proc
        if proc is None:
            return
        self._close_stdin()
        if exc_type is None:
            proc.wait()
            return
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    def write(self, samples: np.ndarray) -> None:
        """Send audio samples to the player.

        Args:
            samples: Float samples in [-1, 1], mono, at ``SAMPLE_RATE``.

        Raises:
            PlayerNotFoundError: If ``ffplay`` is not installed.
            TTSError: If the player stopped unexpectedly.
        """
        payload = to_pcm16(samples)
        if self._proc is None:
            self._start()
            payload = streaming_wav_header() + payload
        try:
            self._proc.stdin.write(payload)
            self._proc.stdin.flush()
        except OSError as exc:
            raise TTSError(
                "ffplay stopped unexpectedly "
                f"(exit code {self._proc.poll()})."
            ) from exc

    def _start(self) -> None:
        """Launch the ``ffplay`` process."""
        try:
            self._proc = subprocess.Popen(
                self.COMMAND, stdin=subprocess.PIPE
            )
        except FileNotFoundError as exc:
            raise PlayerNotFoundError(
                "'ffplay' (from ffmpeg) is not installed."
            ) from exc

    def _close_stdin(self) -> None:
        """Close the pipe to the player, ignoring a broken pipe."""
        stdin = self._proc.stdin
        if stdin is not None:
            with contextlib.suppress(OSError):
                stdin.close()


def require_mp3_encoder() -> None:
    """Check that ``ffmpeg`` is installed and can encode MP3 files.

    Raises:
        TTSError: If ``ffmpeg`` is missing or built without libmp3lame.
    """
    if shutil.which("ffmpeg") is None:
        raise TTSError(
            "'ffmpeg' is required for MP3 output but is not installed."
        )
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    if "libmp3lame" not in result.stdout:
        raise TTSError("This ffmpeg build has no MP3 encoder (libmp3lame).")


class Mp3Writer:
    """Write audio to a variable-bitrate MP3 file through ``ffmpeg``.

    It offers the same ``write`` method and context manager as
    ``soundfile.SoundFile``, so it can replace it when exporting. The
    quality uses LAME's VBR scale (``-q:a``): the encoder picks the bitrate
    of every frame, using less for simple passages such as silences.

    Args:
        path: Destination file.
        quality: VBR quality from 0 (best, largest) to 9 (smallest).
    """

    def __init__(self, path: str, quality: int = DEFAULT_MP3_QUALITY) -> None:
        self._path = path
        self._quality = quality
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> Mp3Writer:
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
            "-i", "pipe:0",
            "-c:a", "libmp3lame", "-q:a", str(self._quality),
            "-f", "mp3", self._path,
        ]
        try:
            self._proc = subprocess.Popen(
                command, stdin=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except FileNotFoundError as exc:
            raise TTSError("'ffmpeg' is not installed.") from exc
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        proc = self._proc
        if proc is None:
            return
        if exc_type is not None:
            self._stop()
            return
        self._proc = None
        _, error = proc.communicate()
        if proc.returncode != 0:
            raise TTSError(
                "ffmpeg failed to encode the MP3 file: "
                f"{error.decode(errors='replace').strip()}"
            )

    def write(self, samples: np.ndarray) -> None:
        """Encode audio samples.

        Args:
            samples: Float samples in [-1, 1], mono, at ``SAMPLE_RATE``.

        Raises:
            TTSError: If ``ffmpeg`` stopped unexpectedly.
        """
        try:
            self._proc.stdin.write(to_pcm16(samples))
        except OSError as exc:
            raise TTSError(
                "ffmpeg stopped unexpectedly while encoding the MP3 "
                f"file: {self._stop()}"
            ) from exc

    def _stop(self) -> str:
        """Stop ffmpeg and return what it printed to standard error."""
        proc, self._proc = self._proc, None
        if proc is None:
            return ""
        proc.kill()
        _, error = proc.communicate()
        return error.decode(errors="replace").strip()


def open_audio_writer(path: str, audio_format: str, mp3_quality: int):
    """Open a file for writing audio incrementally.

    Args:
        path: Destination file.
        audio_format: ``"flac"`` or ``"mp3"``.
        mp3_quality: VBR quality for MP3, from 0 (best) to 9 (smallest).

    Returns:
        A context manager with a ``write`` method that accepts float
        samples in [-1, 1].
    """
    if audio_format == "mp3":
        return Mp3Writer(path, mp3_quality)
    return sf.SoundFile(
        path,
        mode="w",
        samplerate=SAMPLE_RATE,
        channels=1,
        format="FLAC",
        subtype="PCM_16",
    )


def wrap_words(words: list, width: int) -> list:
    """Group word indexes into lines that fit in ``width`` columns.

    Args:
        words: The words of a sentence.
        width: Maximum line length in characters.

    Returns:
        A list of lines, each one a list of indexes into ``words``.
    """
    lines: list = [[]]
    length = 0
    for index, word in enumerate(words):
        extra = len(word) + (1 if lines[-1] else 0)
        if lines[-1] and length + extra > width:
            lines.append([])
            length = 0
            extra = len(word)
        lines[-1].append(index)
        length += extra
    return lines


class KaraokeDisplay:
    """Print sentences with a moving word highlight, without blocking.

    Sentences are queued and shown by a background thread, so speech
    synthesis keeps running ahead of the playback instead of waiting for
    each sentence to be displayed (which would cause audible gaps). When
    standard output is not a terminal, sentences are printed without
    animation.
    """

    PREFIX = "-> "
    HIGHLIGHT = "\033[1;36m"
    RESET = "\033[0m"

    def __init__(self) -> None:
        self._interactive = sys.stdout.isatty()
        self._queue: queue.Queue = queue.Queue()
        self._abort = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, text: str, duration: float) -> None:
        """Queue a sentence to be displayed.

        Args:
            text: The sentence being spoken.
            duration: Duration of its audio, in seconds.
        """
        self._queue.put((text, duration))

    def close(self, abort: bool = False) -> None:
        """Wait for the queued sentences to be shown and stop the thread.

        Args:
            abort: If True, skip the sentences that were not shown yet.
        """
        if abort:
            self._abort.set()
        self._queue.put(None)
        self._thread.join()

    def _run(self) -> None:
        """Show queued sentences until the stop marker arrives."""
        while True:
            item = self._queue.get()
            if item is None:
                return
            if not self._abort.is_set():
                self._show(*item)

    def _show(self, text: str, duration: float) -> None:
        """Highlight each word for a time proportional to its length."""
        words = text.split()
        if not self._interactive or not words:
            print(f"{self.PREFIX}{text}", flush=True)
            return

        columns = shutil.get_terminal_size().columns
        lines = wrap_words(words, max(20, columns - len(self.PREFIX) - 1))
        total_chars = sum(len(word) for word in words)
        for current, word in enumerate(words):
            if self._abort.is_set():
                break
            self._draw(words, lines, current, first=current == 0)
            time.sleep(max(0.05, duration * len(word) / total_chars))
        self._draw(words, lines, None, first=False)

    def _draw(
        self, words: list, lines: list, current: int | None, first: bool
    ) -> None:
        """Redraw the sentence block, highlighting one word."""
        if not first:
            sys.stdout.write(f"\033[{len(lines)}F")  # back to block start
        rows = []
        for number, line in enumerate(lines):
            prefix = self.PREFIX if number == 0 else " " * len(self.PREFIX)
            parts = []
            for index in line:
                if index == current:
                    parts.append(
                        f"{self.HIGHLIGHT}{words[index]}{self.RESET}"
                    )
                else:
                    parts.append(words[index])
            rows.append(prefix + " ".join(parts) + "\033[K")
        sys.stdout.write("\n".join(rows) + "\n")
        sys.stdout.flush()
