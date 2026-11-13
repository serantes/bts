"""Interactive review of pronunciation dictionaries."""

from __future__ import annotations

import os
import re
import stat
import sys
import tempfile
from pathlib import Path

from bts.dictionary import PronunciationDictionary
from bts.errors import TTSError
from bts.synthesis import synthesize_and_play

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


def _speak_entry(line: str, lang_code: str, voice: str | None,
                 speed: float, device: str, max_pause_seconds: float) -> None:
    """Speak the source word using this line's own replacement."""
    content = _LINE_ENDING_RE.sub("", line).lstrip("\ufeff")
    print(content)
    if "|" in content:
        word, replacement = content.split("|", 1)
        word, replacement = word.strip(), replacement.split("|", 1)[0].strip()
        if not word:
            spoken = content.strip()
        else:
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

    synthesize_and_play(
        spoken,
        lang_code=lang_code,
        voice=voice,
        display=False,
        speed=speed,
        device=device,
        max_pause_seconds=max_pause_seconds,
    )


def _read_key(prompt: str) -> str:
    """Read one key without Enter when attached to an interactive terminal."""
    print(prompt, end="", flush=True)
    if not sys.stdin.isatty():
        answer = input("").strip()
    elif os.name == "nt":
        import msvcrt

        answer = msvcrt.getwch()
    else:
        import termios
        import tty

        fd = sys.stdin.fileno()
        previous = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            answer = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, previous)
    print(answer.upper())
    return answer[:1].casefold()


def _decision() -> str:
    """Prompt until the user presses one of the four review keys."""
    choices = {"y": "yes", "n": "no", "r": "repeat", "a": "abort"}
    while True:
        answer = _read_key("Delete? [Y]es, [N]o, [R]epeat, [A]bort: ")
        if answer in choices:
            return choices[answer]
        print("Please press Y, N, R, or A.")


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
    path = Path(filename)
    lines = _read_lines(path)
    marker_index = next(
        (i for i, line in enumerate(lines) if _is_pending_marker(line)),
        None,
    )
    index = marker_index + 1 if marker_index is not None else 0
    changed = False

    while index < len(lines):
        line = lines[index]
        content = _LINE_ENDING_RE.sub("", line).lstrip("\ufeff")
        if not content.strip() or content.lstrip().startswith("#"):
            index += 1
            continue

        while True:
            _speak_entry(
                line,
                lang_code,
                voice,
                speed,
                device,
                max_pause_seconds,
            )
            decision = _decision()
            if decision != "repeat":
                break

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
