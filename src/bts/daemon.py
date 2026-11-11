"""BTS daemon: keeps the Kokoro model loaded for fast repeated invocations.

The daemon listens on a Unix-domain socket and processes synthesis requests
from BTS clients.  The Kokoro pipelines stay in memory between calls,
eliminating the multi-second startup delay on every invocation.

Usage (via CLI)::

    bts daemon start              # start in background
    bts daemon start --fg         # start in foreground (shows logs)
    bts daemon start --timeout 60 # 60-minute inactivity timeout
    bts daemon stop               # stop the running daemon
    bts daemon status             # show daemon status

The daemon shuts itself down after a configurable inactivity period
(default: 30 minutes).  Pass ``--timeout 0`` to disable auto-shutdown.

Wire protocol
-------------
One JSON object per line, UTF-8 encoded.

Client → server (one request per connection)::

    {"action": "play"|"save"|"epub"|"ping"|"status"|"stop", ...params}

Server → client (zero or more progress lines, then exactly one final line)::

    {"status": "progress", "message": "..."}   # forwarded print() output
    {"status": "ok"|"error", "message": "...", "done": true, "exit_code": N}
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import socket
import sys
import threading
from pathlib import Path

from bts.constants import DEFAULT_MP3_QUALITY
from bts.numbers import prepare_text

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_STATE_DIR = Path.home() / ".local" / "share" / "bts"

#: Unix-domain socket used for client–daemon communication.
SOCKET_PATH = _STATE_DIR / "daemon.sock"

#: PID file written while the daemon is alive.
PID_PATH = _STATE_DIR / "daemon.pid"

#: Log file created when the daemon is started in background mode.
LOG_PATH = _STATE_DIR / "daemon.log"

#: Default inactivity timeout (seconds).  The daemon exits after this many
#: seconds without receiving a request.  Zero means "never".
DEFAULT_INACTIVITY_TIMEOUT = 30 * 60


# ---------------------------------------------------------------------------
# Wire-protocol helpers
# ---------------------------------------------------------------------------

def _send(sock: socket.socket, obj: dict) -> None:
    """Send one JSON object followed by a newline over *sock*."""
    data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
    sock.sendall(data)


# ---------------------------------------------------------------------------
# Stdout capture
# ---------------------------------------------------------------------------

class _SocketWriter(io.TextIOBase):
    """Forwards :func:`print` calls to the client as JSON progress messages.

    In-place progress lines (those using ``\\r``) are translated to newlines
    so the client receives clean, separate lines.
    """

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._buf = ""

    def write(self, text: str) -> int:  # type: ignore[override]
        # Treat \r as a line break (overwriting cursor position).
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
        self._buf += cleaned
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            stripped = line.strip()
            if stripped:
                try:
                    _send(
                        self._sock,
                        {"status": "progress", "message": stripped},
                    )
                except OSError:
                    pass
        return len(text)

    def flush(self) -> None:  # type: ignore[override]
        stripped = self._buf.strip()
        if stripped:
            try:
                _send(
                    self._sock,
                    {"status": "progress", "message": stripped},
                )
            except OSError:
                pass
            self._buf = ""


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------

class BTSDaemon:
    """Single-threaded BTS daemon server.

    Connections are served one at a time, which avoids concurrency issues
    with the Kokoro model.  For typical interactive single-user use the
    daemon is never contended.
    """

    def __init__(
        self, inactivity_timeout: int = DEFAULT_INACTIVITY_TIMEOUT
    ) -> None:
        self._timeout = inactivity_timeout
        self._timer: threading.Timer | None = None
        self._running = False
        self._server_sock: socket.socket | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the event loop (blocks until stopped)."""
        _STATE_DIR.mkdir(parents=True, exist_ok=True)

        # Remove any stale socket from a previous (crashed) run.
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()

        self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_sock.bind(str(SOCKET_PATH))
        SOCKET_PATH.chmod(0o700)
        self._server_sock.listen(1)

        PID_PATH.write_text(str(os.getpid()))

        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)

        self._running = True
        self._reset_timer()

        print(
            f"[BTS daemon] PID {os.getpid()} "
            f"listening on {SOCKET_PATH}",
            flush=True,
        )
        if self._timeout > 0:
            print(
                f"[BTS daemon] Inactivity timeout: "
                f"{self._timeout // 60} min",
                flush=True,
            )

        try:
            while self._running:
                self._server_sock.settimeout(1.0)
                try:
                    conn, _ = self._server_sock.accept()
                except socket.timeout:
                    continue
                self._reset_timer()
                try:
                    self._handle_connection(conn)
                finally:
                    with contextlib.suppress(OSError):
                        conn.close()
        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup()

    def _on_signal(self, signum: int, frame: object) -> None:  # noqa: ARG002
        print(
            f"\n[BTS daemon] Signal {signum} — shutting down.", flush=True
        )
        self._running = False

    def _timeout_shutdown(self) -> None:
        print("[BTS daemon] Inactivity timeout — shutting down.", flush=True)
        self._running = False

    def _reset_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        if self._timeout > 0:
            self._timer = threading.Timer(
                self._timeout, self._timeout_shutdown
            )
            self._timer.daemon = True
            self._timer.start()

    def _cleanup(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        if self._server_sock is not None:
            with contextlib.suppress(OSError):
                self._server_sock.close()
        for path in (SOCKET_PATH, PID_PATH):
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
        print("[BTS daemon] Stopped.", flush=True)

    # ------------------------------------------------------------------
    # Connection handling
    # ------------------------------------------------------------------

    def _handle_connection(self, conn: socket.socket) -> None:
        reader = conn.makefile("r", encoding="utf-8")
        try:
            line = reader.readline()
            if not line:
                return
            request = json.loads(line.strip())
        except (json.JSONDecodeError, OSError, ValueError):
            return
        finally:
            # Closing the read wrapper does NOT close conn because conn is
            # still referenced by the caller.
            with contextlib.suppress(OSError):
                reader.close()

        try:
            self._dispatch(request, conn)
        except Exception as exc:  # noqa: BLE001
            with contextlib.suppress(OSError):
                _send(
                    conn,
                    {
                        "status": "error",
                        "message": str(exc),
                        "done": True,
                        "exit_code": 1,
                    },
                )

    def _dispatch(self, request: dict, conn: socket.socket) -> None:
        action = request.get("action", "")

        if action == "ping":
            _send(
                conn,
                {
                    "status": "ok",
                    "message": "pong",
                    "done": True,
                    "exit_code": 0,
                },
            )

        elif action == "stop":
            _send(
                conn,
                {
                    "status": "ok",
                    "message": "Daemon stopping.",
                    "done": True,
                    "exit_code": 0,
                },
            )
            self._running = False

        elif action == "status":
            _send(
                conn,
                {
                    "status": "ok",
                    "message": f"BTS daemon running (PID {os.getpid()})",
                    "done": True,
                    "exit_code": 0,
                },
            )

        elif action == "play":
            self._handle_play(request, conn)

        elif action == "save":
            self._handle_save(request, conn)

        elif action == "epub":
            self._handle_epub(request, conn)

        else:
            _send(
                conn,
                {
                    "status": "error",
                    "message": f"Unknown action: '{action}'",
                    "done": True,
                    "exit_code": 1,
                },
            )

    # ------------------------------------------------------------------
    # Stdout capture context manager
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _capture_stdout(self, conn: socket.socket):
        """Redirect :func:`print` output to the client as progress messages."""
        writer = _SocketWriter(conn)
        old_stdout = sys.stdout
        sys.stdout = writer
        try:
            yield writer
        finally:
            writer.flush()
            sys.stdout = old_stdout

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _build_lexicon(self, request: dict):
        """Return a :class:`PronunciationDictionary` from the request."""
        from bts.dictionary import PronunciationDictionary, load_dictionary

        dict_path = request.get("dictionary")
        if not dict_path:
            return PronunciationDictionary({})
        ignore_case = request.get("dictionary_ignore_case", False)
        return load_dictionary(dict_path, ignore_case)

    def _send_exc(
        self, conn: socket.socket, exc: Exception, exit_code: int = 1
    ) -> None:
        """Send an error response for *exc*."""
        with contextlib.suppress(OSError):
            _send(
                conn,
                {
                    "status": "error",
                    "message": str(exc),
                    "done": True,
                    "exit_code": exit_code,
                },
            )

    # ------------------------------------------------------------------
    # Synthesis handlers
    # ------------------------------------------------------------------

    def _handle_play(self, request: dict, conn: socket.socket) -> None:
        from bts.constants import EXIT_GPU_BUSY
        from bts.errors import GpuMemoryError
        from bts.synthesis import synthesize_and_play

        text = request["text"]
        lang = request.get("lang", "es")
        voice = request.get("voice")
        speed = float(request.get("speed", 1.0))
        device = request.get("device", "cpu")
        lexicon = self._build_lexicon(request)

        with self._capture_stdout(conn):
            try:
                synthesize_and_play(
                    prepare_text(text, lang, lexicon),
                    lang_code=lang,
                    voice=voice,
                    display=False,  # Karaoke requires a foreground process
                    speed=speed,
                    device=device,
                )
                _send(conn, {"status": "ok", "done": True, "exit_code": 0})
            except GpuMemoryError as exc:
                self._send_exc(conn, exc, EXIT_GPU_BUSY)
            except Exception as exc:  # noqa: BLE001
                self._send_exc(conn, exc)

    def _handle_save(self, request: dict, conn: socket.socket) -> None:
        from bts.constants import EXIT_GPU_BUSY
        from bts.errors import GpuMemoryError
        from bts.synthesis import synthesize_text_to_file

        text = request["text"]
        lang = request.get("lang", "es")
        voice = request.get("voice")
        speed = float(request.get("speed", 1.0))
        device = request.get("device", "cpu")
        output = request["output"]
        audio_format = request.get("format", "flac")
        mp3_quality = int(request.get("mp3_quality", DEFAULT_MP3_QUALITY))
        lexicon = self._build_lexicon(request)

        with self._capture_stdout(conn):
            try:
                synthesize_text_to_file(
                    prepare_text(text, lang, lexicon),
                    output,
                    lang_code=lang,
                    voice=voice,
                    speed=speed,
                    device=device,
                    audio_format=audio_format,
                    mp3_quality=mp3_quality,
                )
                _send(conn, {"status": "ok", "done": True, "exit_code": 0})
            except GpuMemoryError as exc:
                self._send_exc(conn, exc, EXIT_GPU_BUSY)
            except Exception as exc:  # noqa: BLE001
                self._send_exc(conn, exc)

    def _handle_epub(self, request: dict, conn: socket.socket) -> None:
        from bts.constants import EXIT_GPU_BUSY
        from bts.epub import process_epub_book
        from bts.errors import GpuMemoryError

        epub_path = request["epub_path"]
        lang = request.get("lang", "es")
        voice = request.get("voice")
        speed = float(request.get("speed", 1.0))
        device = request.get("device", "cpu")
        output = request.get("output")
        audio_format = request.get("format", "flac")
        mp3_quality = int(request.get("mp3_quality", DEFAULT_MP3_QUALITY))
        section_start = request.get("section_start")
        lexicon = self._build_lexicon(request)

        with self._capture_stdout(conn):
            try:
                failed = process_epub_book(
                    epub_path=epub_path,
                    lang_code=lang,
                    voice=voice,
                    output_arg=output,
                    section_list=False,
                    section_start=section_start,
                    display=False,  # Karaoke requires a foreground process
                    speed=speed,
                    device=device,
                    audio_format=audio_format,
                    mp3_quality=mp3_quality,
                    lexicon=lexicon,
                )
                _send(
                    conn,
                    {
                        "status": "ok",
                        "done": True,
                        "exit_code": 1 if failed else 0,
                    },
                )
            except GpuMemoryError as exc:
                self._send_exc(conn, exc, EXIT_GPU_BUSY)
            except Exception as exc:  # noqa: BLE001
                self._send_exc(conn, exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_daemon(inactivity_timeout: int = DEFAULT_INACTIVITY_TIMEOUT) -> None:
    """Create and start the BTS daemon (blocks until stopped)."""
    BTSDaemon(inactivity_timeout=inactivity_timeout).start()


if __name__ == "__main__":
    import argparse

    _p = argparse.ArgumentParser(
        prog="bts-daemon",
        description="BTS daemon — keeps the Kokoro model loaded in memory.",
    )
    _p.add_argument(
        "--timeout",
        type=int,
        default=30,
        metavar="MINUTES",
        help="Inactivity timeout in minutes (0 = never). Default: 30.",
    )
    _opts = _p.parse_args()
    run_daemon(inactivity_timeout=_opts.timeout * 60)
