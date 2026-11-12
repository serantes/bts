"""BTS daemon: keeps the Kokoro model loaded for fast repeated invocations.

The daemon listens on a Unix-domain socket and processes synthesis requests
from BTS clients.  The Kokoro pipelines stay in memory between calls,
eliminating the multi-second startup delay on every invocation.

Usage (via CLI)::

    bts daemon start              # start in background
    bts daemon start --fg         # start in foreground (shows logs)
    bts daemon start --timeout 60 # 60-minute inactivity timeout
    bts daemon pause               # pause the current conversion
    bts daemon play                # resume a paused conversion
    bts daemon resume              # resume a paused conversion
    bts daemon stop                # stop the current conversion
    bts daemon next                # skip the current task
    bts daemon shutdown            # stop the daemon process
    bts daemon status             # show daemon status

The daemon shuts itself down after a configurable inactivity period
(default: 30 minutes).  Pass ``--timeout 0`` to disable auto-shutdown.

Wire protocol
-------------
One JSON object per line, UTF-8 encoded.

Client → server (one request per connection)::

    {"action": "play"|"save"|"epub"|"ping"|"status"|"list"|"delete"|
    "up"|"down"|"move"|"pause"|"resume"|"stop"|"next"|"shutdown",
    ...params}

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
from collections import deque
from pathlib import Path

from bts.constants import DEFAULT_MAX_PAUSE_SECONDS, DEFAULT_MP3_QUALITY
from bts.numbers import prepare_text
from bts.synthesis import SynthesisCancelled

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
    """BTS daemon with a responsive control socket and serial work queue.

    A single worker owns the Kokoro model. Connection handlers only enqueue
    work or answer control requests, so status and control stay responsive.
    """

    def __init__(
        self, inactivity_timeout: int = DEFAULT_INACTIVITY_TIMEOUT
    ) -> None:
        self._timeout = inactivity_timeout
        self._timer: threading.Timer | None = None
        self._running = False
        self._server_sock: socket.socket | None = None
        self._work_queue: deque[tuple[dict, socket.socket]] = deque()
        self._queue_condition = threading.Condition()
        self._active_action: str | None = None
        self._active_request: dict | None = None
        self._paused = False
        self._resume_event = threading.Event()
        self._resume_event.set()
        self._cancel_current = threading.Event()
        self._worker: threading.Thread | None = None

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
        self._server_sock.listen()

        PID_PATH.write_text(str(os.getpid()))

        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)

        self._running = True
        self._worker = threading.Thread(
            target=self._work_loop, name="bts-work-queue", daemon=True
        )
        self._worker.start()
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
                threading.Thread(
                    target=self._handle_connection,
                    args=(conn,),
                    name="bts-client",
                    daemon=True,
                ).start()
        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup()

    def _on_signal(self, signum: int, frame: object) -> None:  # noqa: ARG002
        print(
            f"\n[BTS daemon] Signal {signum} — shutting down.", flush=True
        )
        with self._queue_condition:
            self._running = False
            self._queue_condition.notify_all()

    def _timeout_shutdown(self) -> None:
        print("[BTS daemon] Inactivity timeout — shutting down.", flush=True)
        with self._queue_condition:
            self._running = False
            self._queue_condition.notify_all()

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
        self._resume_event.set()
        self._cancel_current.set()
        with self._queue_condition:
            self._running = False
            pending = list(self._work_queue)
            self._work_queue.clear()
            self._queue_condition.notify_all()
        if self._server_sock is not None:
            with contextlib.suppress(OSError):
                self._server_sock.close()
        if self._worker is not None:
            self._worker.join(timeout=5)
        for _, conn in pending:
            self._send_exc(conn, RuntimeError("Daemon is shutting down."))
            with contextlib.suppress(OSError):
                conn.close()
        for path in (SOCKET_PATH, PID_PATH):
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
        print("[BTS daemon] Stopped.", flush=True)

    # ------------------------------------------------------------------
    # Connection handling
    # ------------------------------------------------------------------

    def _handle_connection(self, conn: socket.socket) -> None:
        owns_connection = False
        try:
            reader = conn.makefile("r", encoding="utf-8")
            try:
                line = reader.readline()
                if not line:
                    return
                request = json.loads(line.strip())
            except (json.JSONDecodeError, OSError, ValueError):
                return
            finally:
                # Closing the read wrapper does NOT close conn because conn
                # is still referenced by the caller or work queue.
                with contextlib.suppress(OSError):
                    reader.close()
            owns_connection = self._dispatch(request, conn)
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
        finally:
            if not owns_connection:
                with contextlib.suppress(OSError):
                    conn.close()

    def _dispatch(self, request: dict, conn: socket.socket) -> bool:
        """Dispatch a request; return True when queued work owns *conn*."""
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

        elif action == "shutdown":
            _send(
                conn,
                {
                    "status": "ok",
                    "message": "Daemon stopping.",
                    "done": True,
                    "exit_code": 0,
                },
            )
            with self._queue_condition:
                self._running = False
                self._queue_condition.notify_all()

        elif action == "status":
            with self._queue_condition:
                active = self._active_action
                paused = self._paused
                queued = len(self._work_queue)
            state = "paused" if paused else "processing" if active else "idle"
            details = f"BTS daemon {state} (PID {os.getpid()})"
            if active:
                details += f"; current request: {active}"
            if queued:
                details += f"; queued requests: {queued}"
            _send(
                conn,
                {
                    "status": "ok",
                    "message": details,
                    "done": True,
                    "exit_code": 0,
                },
            )

        elif action == "list":
            _send(
                conn,
                {
                    "status": "ok",
                    "message": self._format_queue(),
                    "done": True,
                    "exit_code": 0,
                },
            )

        elif action in {"delete", "up", "down", "move"}:
            self._reorder_queue(action, request, conn)

        elif action == "pause":
            with self._queue_condition:
                self._paused = True
                self._resume_event.clear()
            _send(
                conn,
                {
                    "status": "ok",
                    "message": "Processing paused.",
                    "done": True,
                    "exit_code": 0,
                },
            )

        elif action == "resume" or (
            action == "play" and "text" not in request
        ):
            with self._queue_condition:
                was_paused = self._paused
                if was_paused:
                    self._paused = False
                    self._resume_event.set()
                    self._queue_condition.notify_all()
            message = (
                "Processing resumed."
                if was_paused else "No paused processing to resume."
            )
            _send(
                conn,
                {
                    "status": "ok",
                    "message": message,
                    "done": True,
                    "exit_code": 0,
                },
            )

        elif action == "start":
            _send(
                    conn,
                    {
                        "status": "ok",
                        "message": "Daemon is already running.",
                        "done": True,
                        "exit_code": 0,
                    },
            )

        elif action == "stop":
            with self._queue_condition:
                active = self._active_action is not None
                if active:
                    self._cancel_current.set()
            message = (
                "Stopping current request."
                if active else "No request is currently processing."
            )
            _send(
                conn,
                {
                    "status": "ok",
                    "message": message,
                    "done": True,
                    "exit_code": 0,
                },
            )

        elif action == "next":
            with self._queue_condition:
                active = self._active_action is not None
                queued = bool(self._work_queue)
                if active:
                    self._cancel_current.set()
                if self._paused:
                    self._paused = False
                    self._resume_event.set()
                    self._queue_condition.notify_all()

            if active and queued:
                message = "Current task skipped; moving to the next task."
            elif active:
                message = "Current task skipped; no next task is queued."
            elif queued:
                message = "Continuing with the next queued task."
            else:
                message = "No task is currently processing."
            _send(
                conn,
                {
                    "status": "ok",
                    "message": message,
                    "done": True,
                    "exit_code": 0,
                },
            )

        elif action == "save":
            return self._enqueue(request, conn)
        elif action == "epub":
            return self._enqueue(request, conn)
        elif action == "play":
            return self._enqueue(request, conn)
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
        return False

    def _enqueue(self, request: dict, conn: socket.socket) -> bool:
        with self._queue_condition:
            self._work_queue.append((request, conn))
            position = len(self._work_queue)
            self._queue_condition.notify()
        _send(
            conn,
            {
                "status": "progress",
                "message": f"Request queued (position {position}).",
            },
        )
        return True

    @staticmethod
    def _task_description(request: dict) -> str:
        action = request.get("action", "request")
        if action == "epub":
            detail = Path(str(request.get("epub_path", ""))).name
        elif action == "save":
            detail = Path(str(request.get("output", ""))).name
        else:
            detail = " ".join(str(request.get("text", "")).split())[:60]
        return f"{action}: {detail}" if detail else action

    def _format_queue(self) -> str:
        with self._queue_condition:
            active = self._active_request
            pending = list(self._work_queue)
        if active is None and not pending:
            return "The queue is empty."

        lines = ["Task queue:"]
        position = 1
        if active is not None:
            lines.append(
                f"  {position}. [running] {self._task_description(active)}"
            )
            position += 1
        for request, _ in pending:
            lines.append(
                f"  {position}. [pending] {self._task_description(request)}"
            )
            position += 1
        return "\n".join(lines)

    def _reorder_queue(
        self, action: str, request: dict, conn: socket.socket
    ) -> None:
        fields = ("number",) if action in {"delete", "up", "down"} else (
            "from", "to"
        )
        values: list[int] = []
        for field in fields:
            value = request.get(field)
            if isinstance(value, bool) or not isinstance(value, int):
                self._send_exc(
                    conn, ValueError(f"'{field}' must be a positive integer.")
                )
                return
            if value < 1:
                self._send_exc(
                    conn, ValueError(f"'{field}' must be a positive integer.")
                )
                return
            values.append(value)

        removed: tuple[dict, socket.socket] | None = None
        with self._queue_condition:
            active_offset = 1 if self._active_request is not None else 0
            pending = list(self._work_queue)
            total = active_offset + len(pending)

            if action == "delete":
                number = values[0]
                if number <= active_offset:
                    error = "The running task cannot be deleted."
                elif number > total:
                    error = f"No task exists at position {number}."
                else:
                    removed = pending.pop(number - active_offset - 1)
                    self._work_queue = deque(pending)
                    error = ""
                message = f"Task {number} deleted."

            elif action in {"up", "down"}:
                number = values[0]
                if number <= active_offset:
                    error = "The running task cannot be moved."
                elif number > total:
                    error = f"No task exists at position {number}."
                else:
                    index = number - active_offset - 1
                    neighbor = index - 1 if action == "up" else index + 1
                    if neighbor < 0 or neighbor >= len(pending):
                        message = (
                            f"Task {number} is already at the "
                            f"{'front' if action == 'up' else 'end'} "
                            "of the queue."
                        )
                    else:
                        pending[index], pending[neighbor] = (
                            pending[neighbor], pending[index]
                        )
                        self._work_queue = deque(pending)
                        message = f"Task moved {action}."
                    error = ""

            else:
                source, destination = values
                if source <= active_offset:
                    error = "The running task cannot be moved."
                elif source > total:
                    error = f"No task exists at position {source}."
                elif not pending:
                    error = "There are no pending tasks to move."
                else:
                    first_pending = active_offset + 1
                    target = min(max(destination, first_pending), total)
                    item = pending.pop(source - active_offset - 1)
                    pending.insert(target - active_offset - 1, item)
                    self._work_queue = deque(pending)
                    error = ""
                    message = (
                        f"Task {source} moved to position {target}."
                    )

        if removed is not None:
            _, task_conn = removed
            with contextlib.suppress(OSError):
                _send(
                    task_conn,
                    {
                        "status": "error",
                        "message": "Request removed from the queue.",
                        "done": True,
                        "exit_code": 1,
                    },
                )
            with contextlib.suppress(OSError):
                task_conn.close()

        if error:
            self._send_exc(conn, ValueError(error))
        else:
            _send(
                conn,
                {
                    "status": "ok",
                    "message": message,
                    "done": True,
                    "exit_code": 0,
                },
            )

    def _checkpoint(self) -> None:
        """Wait through a pause or raise when the active request is stopped."""
        while True:
            if self._cancel_current.is_set():
                raise SynthesisCancelled("Request stopped.")
            if self._resume_event.wait(timeout=0.1):
                break
        if self._cancel_current.is_set():
            raise SynthesisCancelled("Request stopped.")

    def _work_loop(self) -> None:
        while True:
            with self._queue_condition:
                while (
                    self._running
                    and (self._paused or not self._work_queue)
                ):
                    self._queue_condition.wait()
                if not self._running:
                    return
                request, conn = self._work_queue.popleft()
                action = request.get("action", "")
                self._active_action = action
                self._active_request = request
                self._cancel_current.clear()
            try:
                self._checkpoint()
                with self._capture_stdout(conn):
                    if action == "play":
                        self._handle_play(request, conn)
                    elif action == "save":
                        self._handle_save(request, conn)
                    else:
                        self._handle_epub(request, conn)
            except SynthesisCancelled:
                with contextlib.suppress(OSError):
                    _send(
                        conn,
                        {
                            "status": "ok",
                            "message": "Request stopped.",
                            "done": True,
                            "exit_code": 0,
                        },
                    )
            except Exception as exc:  # noqa: BLE001
                self._send_exc(conn, exc)
            finally:
                with self._queue_condition:
                    self._active_action = None
                    self._active_request = None
                    self._queue_condition.notify_all()
                with contextlib.suppress(OSError):
                    conn.close()

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
        from bts.device import resolve_device
        from bts.errors import GpuMemoryError
        from bts.synthesis import synthesize_and_play

        text = request["text"]
        lang = request.get("lang", "es")
        voice = request.get("voice")
        speed = float(request.get("speed", 1.0))
        max_pause = float(
            request.get("max_pause", DEFAULT_MAX_PAUSE_SECONDS)
        )
        requested_device = request.get("device", "auto")
        lexicon = self._build_lexicon(request)

        with self._capture_stdout(conn):
            try:
                device = resolve_device(requested_device)
                synthesize_and_play(
                    prepare_text(text, lang, lexicon),
                    lang_code=lang,
                    voice=voice,
                    display=False,  # Karaoke requires a foreground process
                    speed=speed,
                    max_pause_seconds=max_pause,
                    device=device,
                    checkpoint=self._checkpoint,
                )
                _send(conn, {"status": "ok", "done": True, "exit_code": 0})
            except GpuMemoryError as exc:
                self._send_exc(conn, exc, EXIT_GPU_BUSY)
            except SynthesisCancelled:
                raise
            except Exception as exc:  # noqa: BLE001
                self._send_exc(conn, exc)

    def _handle_save(self, request: dict, conn: socket.socket) -> None:
        from bts.constants import EXIT_GPU_BUSY
        from bts.device import resolve_device
        from bts.errors import GpuMemoryError
        from bts.synthesis import synthesize_text_to_file

        text = request["text"]
        lang = request.get("lang", "es")
        voice = request.get("voice")
        speed = float(request.get("speed", 1.0))
        max_pause = float(
            request.get("max_pause", DEFAULT_MAX_PAUSE_SECONDS)
        )
        requested_device = request.get("device", "auto")
        output = request["output"]
        audio_format = request.get("format", "flac")
        mp3_quality = int(request.get("mp3_quality", DEFAULT_MP3_QUALITY))
        lexicon = self._build_lexicon(request)

        with self._capture_stdout(conn):
            try:
                device = resolve_device(requested_device)
                synthesize_text_to_file(
                    prepare_text(text, lang, lexicon),
                    output,
                    lang_code=lang,
                    voice=voice,
                    speed=speed,
                    max_pause_seconds=max_pause,
                    device=device,
                    audio_format=audio_format,
                    mp3_quality=mp3_quality,
                    checkpoint=self._checkpoint,
                )
                _send(conn, {"status": "ok", "done": True, "exit_code": 0})
            except GpuMemoryError as exc:
                self._send_exc(conn, exc, EXIT_GPU_BUSY)
            except SynthesisCancelled:
                raise
            except Exception as exc:  # noqa: BLE001
                self._send_exc(conn, exc)

    def _handle_epub(self, request: dict, conn: socket.socket) -> None:
        from bts.constants import EXIT_GPU_BUSY
        from bts.device import resolve_device
        from bts.epub import process_epub_book
        from bts.errors import GpuMemoryError

        epub_path = request["epub_path"]
        lang = request.get("lang", "es")
        voice = request.get("voice")
        speed = float(request.get("speed", 1.0))
        max_pause = float(
            request.get("max_pause", DEFAULT_MAX_PAUSE_SECONDS)
        )
        requested_device = request.get("device", "auto")
        output = request.get("output")
        audio_format = request.get("format", "flac")
        mp3_quality = int(request.get("mp3_quality", DEFAULT_MP3_QUALITY))
        section_start = request.get("section_start")
        lexicon = self._build_lexicon(request)

        with self._capture_stdout(conn):
            try:
                device = resolve_device(requested_device)
                failed = process_epub_book(
                    epub_path=epub_path,
                    lang_code=lang,
                    voice=voice,
                    output_arg=output,
                    sections_list=False,
                    section_start=section_start,
                    display=False,  # Karaoke requires a foreground process
                    speed=speed,
                    max_pause_seconds=max_pause,
                    device=device,
                    audio_format=audio_format,
                    mp3_quality=mp3_quality,
                    lexicon=lexicon,
                    checkpoint=self._checkpoint,
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
            except SynthesisCancelled:
                raise
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
