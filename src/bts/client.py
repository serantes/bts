"""Lightweight BTS daemon client.

Connects to the BTS daemon over a Unix-domain socket and forwards synthesis
requests.  If the daemon is not running, every public function returns
``None`` or ``False`` so the caller can fall back to standalone mode
without any visible delay.

The connection timeout is kept intentionally short (0.5 s) so a missing or
unresponsive daemon never adds noticeable latency to standalone invocations.
"""

from __future__ import annotations

import json
import socket
import sys

# SOCKET_PATH is defined in daemon.py so both sides always agree on the path.
from bts.daemon import SOCKET_PATH

#: Maximum time to wait when connecting to the daemon (seconds).
#: Kept short so a missing daemon never delays the CLI noticeably.
CONNECT_TIMEOUT = 0.5


# ---------------------------------------------------------------------------
# Wire-protocol helpers
# ---------------------------------------------------------------------------

def _send(sock: socket.socket, obj: dict) -> None:
    data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
    sock.sendall(data)


def _iter_responses(sock: socket.socket):
    """Yield JSON response objects until the final ``"done": true`` message."""
    reader = sock.makefile("r", encoding="utf-8")
    try:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield obj
            if obj.get("done"):
                break
    finally:
        try:
            reader.close()
        except OSError:
            pass


def _connect() -> socket.socket | None:
    """Return a connected socket to the daemon, or ``None`` if unavailable."""
    if not SOCKET_PATH.exists():
        return None
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(CONNECT_TIMEOUT)
        sock.connect(str(SOCKET_PATH))
        # Once connected, disable timeout: synthesis can take many minutes.
        sock.settimeout(None)
        return sock
    except (OSError, socket.timeout):
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def try_daemon_request(request: dict) -> int | None:
    """Send *request* to the daemon and return the exit code.

    Returns ``None`` if the daemon is not running, so the caller can fall
    back to standalone mode transparently.

    Progress lines printed by the daemon are forwarded to the client's
    ``stdout`` in real time, giving the user the same output they would see
    in standalone mode.

    Args:
        request: JSON-serialisable dict with at least an ``"action"`` key.

    Returns:
        Exit code (int) when the daemon handled the request, or ``None`` if
        the daemon was unavailable.
    """
    sock = _connect()
    if sock is None:
        return None

    try:
        _send(sock, request)
        exit_code = 1
        for msg in _iter_responses(sock):
            status = msg.get("status", "")
            message = msg.get("message", "")
            if status == "progress" and message:
                print(message, flush=True)
            elif status == "error" and message:
                print(f"Error: {message}", file=sys.stderr, flush=True)
            if "exit_code" in msg:
                exit_code = int(msg["exit_code"])
        return exit_code
    except OSError:
        return None
    finally:
        try:
            sock.close()
        except OSError:
            pass


def ping_daemon() -> bool:
    """Return ``True`` if the daemon is running and responsive."""
    return try_daemon_request({"action": "ping"}) == 0


def stop_daemon() -> bool:
    """Ask the daemon process to shut down.

    Returns:
        ``True`` if the daemon was running and acknowledged the request.
    """
    return try_daemon_request({"action": "shutdown"}) is not None


def control_daemon(
    action: str, **parameters: int
) -> tuple[str, bool] | None:
    """Send a control action and return its message and success state."""
    sock = _connect()
    if sock is None:
        return None
    try:
        _send(sock, {"action": action, **parameters})
        for msg in _iter_responses(sock):
            if msg.get("done"):
                return (
                    str(msg.get("message", "")),
                    msg.get("status") != "error",
                )
    except OSError:
        return None
    finally:
        try:
            sock.close()
        except OSError:
            pass
    return None


def get_daemon_status() -> str | None:
    """Return the daemon's status string, or ``None`` if not running."""
    sock = _connect()
    if sock is None:
        return None
    try:
        _send(sock, {"action": "status"})
        for msg in _iter_responses(sock):
            if msg.get("message"):
                return msg["message"]
    except OSError:
        pass
    finally:
        try:
            sock.close()
        except OSError:
            pass
    return None
