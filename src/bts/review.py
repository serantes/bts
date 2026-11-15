"""Interactive review of pronunciation dictionaries."""

from __future__ import annotations

import contextlib
import os
import re
import select
import shutil
import stat
import sys
import tempfile
import threading
import unicodedata
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path

from bts.device import get_pipeline
from bts.dictionary import PronunciationDictionary
from bts.errors import TTSError
from bts.languages import SUPPORTED_LANGUAGES
from bts.numbers import prepare_text
from bts.synthesis import SynthesisCancelled, synthesize_and_play

_PENDING_MARKER = "# BTS REVIEW PENDING"
_LINE_ENDING_RE = re.compile(r"[\r\n]+$")


def _write_lines(path: Path, lines: list[str]) -> None:
    """Atomically replace the file while preserving its permissions."""
    temporary_path = None
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.writelines(lines)
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    except OSError as exc:
        raise TTSError(f"Cannot update dictionary '{path}': {exc}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _read_lines(path: Path) -> list[str]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return handle.readlines()
    except OSError as exc:
        raise TTSError(f"Cannot read dictionary '{path}': {exc}") from exc
    except UnicodeDecodeError as exc:
        raise TTSError(
            f"Dictionary '{path}' is not valid UTF-8: {exc}"
        ) from exc


def _is_pending_marker(line: str) -> bool:
    return _LINE_ENDING_RE.sub("", line) == _PENDING_MARKER


def _speak_entry(
    line: str, lang_code: str, voice: str | None, speed: float, device: str,
    max_pause_seconds: float, *, cancel_event: threading.Event | None = None,
    show: bool = True,
) -> None:
    """Speak the source word, then its dictionary pronunciation."""
    content = _LINE_ENDING_RE.sub("", line).lstrip("\ufeff")
    if show:
        print(content)
    title = ""
    if "|" in content:
        word, replacement = content.split("|", 1)
        word, replacement = word.strip(), replacement.split("|", 1)[0].strip()
        if not word:
            spoken = content.strip()
        else:
            title = prepare_text(word, lang_code, PronunciationDictionary({}))
            try:
                spoken = PronunciationDictionary(
                    {word: replacement}
                ).apply(word)
            except ValueError as exc:
                raise TTSError(
                    f"Invalid pronunciation entry '{content}': {exc}"
                ) from exc
    else:
        spoken = content.strip()

    cancellation = {"cancel_event": cancel_event} if cancel_event else {}
    synthesize_and_play(
        spoken,
        lang_code=lang_code,
        voice=voice,
        display=False,
        speed=speed,
        device=device,
        title=title,
        title_speed_factor=1.0,
        max_pause_seconds=max_pause_seconds,
        show_player_status=False,
        **cancellation,
    )


def _read_key(prompt: str) -> str:
    """Read one key without Enter when attached to an interactive terminal."""
    print(prompt, end="", flush=True)
    if not sys.stdin.isatty():
        answer = input("").strip()
    else:
        with _terminal():
            answer = _read_terminal_key()
    print(answer.upper())
    return answer[:1].casefold()


def _decision() -> str:
    """Prompt until the user presses a review key."""
    while True:
        answer = _read_key(_PROMPT)
        if answer in _CHOICES:
            return _CHOICES[answer]
        print("Please press Y, N, P, E, U, R, or A.")


_PROMPT = (
    "Delete? [Y]es, [N]o, [P]lay again, [E]dit, "
    "[U]ndo, [R]edo, [A]bort: "
)
_CHOICES = {
    "y": "yes", "n": "no", "p": "repeat", "e": "edit",
    "u": "undo", "r": "redo", "a": "abort",
}
_KEY_BUFFER: deque[str] = deque()


@contextlib.contextmanager
def _terminal():
    if os.name == "nt":
        yield
        return
    import termios
    import tty

    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd, termios.TCSANOW)
        settings = termios.tcgetattr(fd)
        settings[6][termios.VSUSP] = b"\0"
        termios.tcsetattr(fd, termios.TCSANOW, settings)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def _key_ready(timeout: float) -> bool:
    if _KEY_BUFFER:
        return True
    if os.name == "nt":
        import msvcrt
        import time

        if not msvcrt.kbhit():
            time.sleep(timeout)
        return msvcrt.kbhit()
    return bool(select.select([sys.stdin.fileno()], [], [], timeout)[0])


def _read_terminal_key() -> str:
    if _KEY_BUFFER:
        return _KEY_BUFFER.popleft()
    if os.name == "nt":
        import msvcrt

        key = msvcrt.getwch()
        if key == "\x03":
            raise KeyboardInterrupt
        if key in {"\x00", "\xe0"}:
            return {"K": "left", "M": "right", "G": "home", "O": "end",
                    "S": "delete"}.get(msvcrt.getwch(), "")
        return key

    import codecs

    fd = sys.stdin.fileno()
    decoder = codecs.getincrementaldecoder(sys.stdin.encoding or "utf-8")()

    def character() -> str:
        while True:
            data = os.read(fd, 1)
            if not data:
                raise EOFError("Terminal input closed")
            value = decoder.decode(data)
            if value:
                return value

    key = character()
    if key != "\x1b" or not _key_ready(0.05):
        return key
    sequence = character()
    if sequence not in {"[", "O"}:
        _KEY_BUFFER.append(sequence)
        return "\x1b"
    while _key_ready(0.05):
        char = character()
        sequence += char
        if char.isalpha() or char == "~":
            break
    return {
        "[D": "left", "[C": "right", "[H": "home", "[F": "end",
        "OH": "home", "OF": "end", "[1~": "home", "[4~": "end",
        "[3~": "delete", "[122;6u": "redo", "[90;6u": "redo",
        "[27;6;122~": "redo", "[27;6;90~": "redo",
        "[122;5u": "\x1a", "[90;5u": "\x1a",
        "[27;5;122~": "\x1a", "[27;5;90~": "\x1a",
        "[121;5u": "\x19", "[27;5;121~": "\x19",
        "[27;5;89~": "\x19", "[27u": "\x1b", "[13u": "\r",
    }.get(sequence, "")


def _width(text: str) -> int:
    return sum(
        0 if unicodedata.combining(char)
        else 2 if unicodedata.east_asian_width(char) in {"W", "F"}
        else 1
        for char in text
    )


def _edit_entry(content: str) -> str:
    if not sys.stdin.isatty():
        value = input("Edit (Esc to cancel): ")
        return content if value == "\x1b" else value

    text = content
    cursor = len(text)
    undo: list[tuple[str, int]] = []
    redo: list[tuple[str, int]] = []
    prefix = "Edit: "
    print("Edit: Enter accepts, Esc cancels, Ctrl+Z undoes, "
          "Ctrl+Shift+Z or Ctrl+Y redoes.")
    with _terminal():
        # Request enhanced key reporting where the terminal supports it.
        sys.stdout.write("\033[>1u\033[>4;2m")
        try:
            while True:
                available = max(1, shutil.get_terminal_size().columns
                                - len(prefix) - 1)
                start = cursor
                while (
                    start > 0
                    and _width(text[start - 1:cursor]) <= available
                ):
                    start -= 1
                end = cursor
                while (
                    end < len(text)
                    and _width(text[start:end + 1]) <= available
                ):
                    end += 1
                sys.stdout.write(
                    f"\r\033[K{prefix}{text[start:end]}\r"
                    f"\033[{len(prefix) + _width(text[start:cursor])}C"
                )
                sys.stdout.flush()
                key = _read_terminal_key()
                if key in {"\r", "\n"}:
                    print()
                    return text
                if key == "\x1b":
                    print()
                    return content
                if key == "left":
                    cursor = max(0, cursor - 1)
                elif key == "right":
                    cursor = min(len(text), cursor + 1)
                elif key == "home":
                    cursor = 0
                elif key == "end":
                    cursor = len(text)
                elif key in {"\x1a", "\x19", "redo"}:
                    source, target = (
                        (undo, redo) if key == "\x1a" else (redo, undo)
                    )
                    if source:
                        target.append((text, cursor))
                        text, cursor = source.pop()
                elif key in {"\x7f", "\x08", "delete"}:
                    position = cursor if key == "delete" else cursor - 1
                    if 0 <= position < len(text):
                        undo.append((text, cursor))
                        redo.clear()
                        text = text[:position] + text[position + 1:]
                        cursor = position
                elif len(key) == 1 and key.isprintable():
                    undo.append((text, cursor))
                    redo.clear()
                    text = text[:cursor] + key + text[cursor:]
                    cursor += 1
        finally:
            sys.stdout.write("\033[<u\033[>4m")
            sys.stdout.flush()


class _SpeechWorker:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending: list[tuple[Future, threading.Event]] = []

    def __enter__(self) -> _SpeechWorker:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        for _, cancel in self._pending:
            cancel.set()
        self._executor.shutdown(wait=True)
        if exc_type is None:
            self.check()

    def start(self, line: str, args: tuple) -> threading.Event:
        cancel = threading.Event()

        def speak() -> None:
            if not cancel.is_set():
                _speak_entry(line, *args, cancel_event=cancel, show=False)

        context = copy_context()
        self._pending.append(
            (self._executor.submit(context.run, speak), cancel)
        )
        return cancel

    def check(self) -> None:
        pending = []
        for future, cancel in self._pending:
            if not future.done():
                pending.append((future, cancel))
                continue
            try:
                future.result()
            except SynthesisCancelled:
                if not cancel.is_set():
                    raise
        self._pending = pending


def _speak_and_decide(line: str, *args, worker: _SpeechWorker) -> str:
    if not sys.stdin.isatty():
        _speak_entry(line, *args)
        return _decision()

    print(_LINE_ENDING_RE.sub("", line).lstrip("\ufeff"), flush=True)
    with _terminal():
        cancel = worker.start(line, args)
        print(_PROMPT, end="", flush=True)
        try:
            while True:
                worker.check()
                if not _key_ready(0.05):
                    continue
                key = _read_terminal_key().casefold()
                if key in _CHOICES:
                    print(key.upper())
                    cancel.set()
                    worker.check()
                    return _CHOICES[key]
                print("\nPlease press Y, N, P, E, U, R, or A.")
                print(_PROMPT, end="", flush=True)
        finally:
            cancel.set()


def _confirm_save() -> bool:
    """Ask whether the edits made during a completed review should be saved."""
    while True:
        answer = _read_key("Save changes? [Y/N]: ")
        if answer in {"y", "n"}:
            return answer == "y"
        print("Please press Y or N.")


def review_dictionary(
    filename: str,
    *,
    lang_code: str,
    voice: str | None,
    speed: float,
    device: str,
    max_pause_seconds: float,
) -> int:
    """Speak and interactively review each pronunciation dictionary entry."""
    with _SpeechWorker() as worker:
        return _review_dictionary(
            filename, lang_code, voice, speed, device, max_pause_seconds,
            worker,
        )


def _review_dictionary(
    filename: str, lang_code: str, voice: str | None, speed: float,
    device: str, max_pause_seconds: float, worker: _SpeechWorker,
) -> int:
    path = Path(filename)
    lines = _read_lines(path)
    marker_index = next(
        (i for i, line in enumerate(lines) if _is_pending_marker(line)),
        None,
    )
    index = marker_index + 1 if marker_index is not None else 0
    changed = False
    prepared = False
    undo_stack: list[tuple[list[str], int, int | None, bool]] = []
    redo_stack: list[tuple[list[str], int, int | None, bool]] = []

    def snapshot() -> tuple[list[str], int, int | None, bool]:
        return lines.copy(), index, marker_index, changed

    def restore(state: tuple[list[str], int, int | None, bool]) -> None:
        nonlocal index, marker_index, changed
        saved_lines, index, marker_index, changed = state
        lines[:] = saved_lines

    initial_state = snapshot()

    while index < len(lines):
        line = lines[index]
        content = _LINE_ENDING_RE.sub("", line).lstrip("\ufeff")
        if not content.strip() or content.lstrip().startswith("#"):
            index += 1
            continue

        if not prepared and sys.stdin.isatty():
            lang_info = SUPPORTED_LANGUAGES.get(lang_code.lower())
            if not lang_info:
                raise TTSError(
                    f"Unsupported language code '{lang_code}'. "
                    "Use --list-languages to view available options."
                )
            print("[INFO] Loading speech model...", flush=True)
            get_pipeline(lang_info["internal"], device)
            prepared = True

        while True:
            decision = _speak_and_decide(
                line,
                lang_code,
                voice,
                speed,
                device,
                max_pause_seconds,
                worker=worker,
            )
            while decision == "edit":
                edited = _edit_entry(content)
                if edited != content:
                    undo_stack.append(snapshot())
                    redo_stack.clear()
                    ending = line[len(_LINE_ENDING_RE.sub("", line)):]
                    bom = "\ufeff" if line.startswith("\ufeff") else ""
                    line = bom + edited + ending
                    lines[index] = line
                    content = edited
                    changed = True
                print(content)
                decision = _decision()
            if decision != "repeat":
                break

        if decision in {"undo", "redo"}:
            if decision == "undo":
                if undo_stack:
                    redo_stack.append(snapshot())
                    restore(undo_stack.pop())
                else:
                    restore(initial_state)
            elif redo_stack:
                undo_stack.append(snapshot())
                restore(redo_stack.pop())
            continue

        if decision == "abort":
            if marker_index is None:
                ending = "\r\n" if line.endswith("\r\n") else "\n"
                lines.insert(index, _PENDING_MARKER + ending)
            print(f"[INFO] Review paused. Resume at '{content}'.")
            if _confirm_save():
                _write_lines(path, lines)
                print("[INFO] Dictionary changes and resume point saved.")
            else:
                print("[INFO] Dictionary changes discarded.")
            return 0

        undo_stack.append(snapshot())
        redo_stack.clear()

        if marker_index is not None:
            del lines[marker_index]
            index -= 1
            marker_index = None
            changed = True

        if decision == "yes":
            del lines[index]
            changed = True
        else:
            index += 1

    if marker_index is not None:
        del lines[marker_index]
        changed = True
    print(f"[INFO] Dictionary review complete: '{path}'.")
    if _confirm_save():
        if changed:
            _write_lines(path, lines)
        print("[INFO] Dictionary changes saved.")
    else:
        print("[INFO] Dictionary changes discarded.")
    return 0
