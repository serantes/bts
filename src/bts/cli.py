"""Kokoro-82M local neural text-to-speech (TTS) client.

Generates speech locally with the Kokoro-82M model and either streams it to
``ffplay`` or exports the sections of an EPUB book as FLAC files.

Features:
    * Text from the command line, from standard input or from a text file
      (``--text``), read aloud or saved as an audio file (``--output``).
    * Optional karaoke-style word highlighting synchronized with the audio.
    * EPUB books: read them aloud (streaming) or export one audio file per
      section, as FLAC or as variable-bitrate MP3 (``--format``). An
      interrupted export can be resumed: finished sections are skipped.
    * Pronunciation dictionary (``--dictionary``): a text file whose lines
      are ``word|replacement``, used to fix mispronounced words. A
      replacement between square brackets is read as phonemes, for example
      ``mr|[mˈɪstəɹ]``.
    * Spelling analysis (``--analyze``): lists the words of a book or text
      that the Hunspell dictionary of the language does not know, ready to
      be used as a pronunciation dictionary. With ``--ipa`` every word also
      gets the phonemes that Kokoro would use, to correct them by hand;
      ``--ipa LANG`` generates them with another language than ``--lang``.
    * Automatically selects the GPU when it has enough free memory, otherwise
      uses the CPU. The choice can be overridden with ``--device``.

Requirements:
    pip install kokoro soundfile numpy torch
    pip install ebooklib beautifulsoup4    # optional, EPUB support only
    ``ffplay`` (part of FFmpeg) must be installed for playback, and
    ``ffmpeg`` built with libmp3lame for MP3 export. ``--analyze`` needs
    ``libhunspell`` and the Hunspell dictionary of the language.

Examples:
    bts "Hola, esto es una prueba."
    bts --epub book.epub
    bts --epub book.epub --output out_dir
    bts --epub book.epub --output --format mp3
    bts --epub book.epub --dictionary words.txt
    bts --text story.txt --output story.mp3
    bts "Hola" --output hola.flac
    bts --epub book.epub --analyze
    bts --epub book.epub --analyze --ipa
    bts --epub book.epub --analyze --lang es --ipa us
    bts --clipboard


Exit status:
    0 on success, when the user presses Ctrl+C, or when the GPU has no free
    memory (see ``EXIT_GPU_BUSY``); 1 on any other error.
"""


from __future__ import annotations

import argparse
import math
import os
import select
import sys

import argcomplete
from argcomplete.completers import ChoicesCompleter

from bts.analysis import run_analysis
from bts.audio import require_ffplay, require_mp3_encoder
from bts.constants import (
    AUDIO_FORMATS,
    DEFAULT_MAX_PAUSE_SECONDS,
    DEFAULT_MP3_QUALITY,
    EXIT_GPU_BUSY,
)
from bts.device import resolve_device
from bts.dictionary import PronunciationDictionary, load_dictionary
from bts.epub import process_epub_book
from bts.errors import GpuMemoryError, TTSError
from bts.languages import (
    KNOWN_VOICES,
    SUPPORTED_LANGUAGES,
    show_languages,
    show_voices,
    validate_voice,
)
from bts.numbers import prepare_text
from bts.project import build_project, create_project
from bts.synthesis import (
    run_voice_tests,
    synthesize_and_play,
    synthesize_text_to_file,
)
from bts.textio import read_text_file, resolve_audio_output


def positive_float(value: str) -> float:
    """Parse a strictly positive float for argparse."""
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'{value}' is not a number"
        ) from None
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("the value must be greater than 0")
    return number


def non_negative_float(value: str) -> float:
    """Parse a finite, non-negative float for argparse."""
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'{value}' is not a number"
        ) from None
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError(
            "the value must be finite and at least 0"
        )
    return number


class _StoreExplicit(argparse.Action):
    """Store an option value and mark that the user supplied it."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, f"{self.dest}_explicit", True)


def _package_version() -> str:
    """Return the installed version of BTS."""
    from bts import __version__

    return __version__


def build_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Kokoro-82M Local Neural Text-to-Speech client. "
            "Converts text or EPUB books into natural speech "
            "using local inference."
        ),
        epilog=(
            "Examples:\n"
            '  bts "Hola, esto es una prueba."\n'
            "  bts --epub libro.epub"
            "                  # Reads directly via streaming\n"
            "  bts --epub libro.epub --output"
            "         # Exports to folder named after EPUB\n"
            "\n"
            "Daemon commands:\n"
            "  bts daemon start             Start the background daemon\n"
            "  bts daemon status            Show daemon status\n"
            "  bts daemon pause             Pause the current conversion\n"
            "  bts daemon play|resume       Resume a paused conversion\n"
            "  bts daemon stop              Stop the current conversion\n"
            "  bts daemon next              Skip to the next queued task\n"
            "  bts daemon list              List active and pending tasks\n"
            "  bts daemon delete <number>   Delete a pending task\n"
            "  bts daemon up|down <number>  Move a pending task in the queue\n"
            "  bts daemon move <from> <to>  Move a pending task to a"
            " position\n"
            "  bts daemon shutdown          Shut down the daemon\n"
            "Run 'bts daemon help' for daemon command details.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.set_defaults(
        lang_explicit=False,
        voice_explicit=False,
        format_explicit=False,
        dictionary_explicit=False,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"BTS - Book To Speech {_package_version()}",
    )
    parser.add_argument(
        "text",
        nargs="?",
        help=(
            "The text to convert to speech. If omitted, reads from "
            "standard input or EPUB."
        ),
    )
    parser.add_argument(
        "--lang",
        action=_StoreExplicit,
        default="es",
        type=str.lower,
        choices=sorted(SUPPORTED_LANGUAGES),
        help="Language code. Default: 'es'.",
    )
    voice_action = parser.add_argument(
        "--voice",
        action=_StoreExplicit,
        default=None,
        help=(
            "Neural voice ID to use. If omitted, uses the first verified "
            "voice for the selected language."
        ),
    )
    voice_action.completer = ChoicesCompleter(sorted(KNOWN_VOICES))
    parser.add_argument(
        "--speed",
        type=positive_float,
        default=1.0,
        help="Speech speed multiplier (default: 1.0).",
    )
    parser.add_argument(
        "--max-pause",
        type=non_negative_float,
        default=DEFAULT_MAX_PAUSE_SECONDS,
        metavar="SECONDS",
        help=(
            "Maximum duration of a text-marker pause in seconds "
            f"(default: {DEFAULT_MAX_PAUSE_SECONDS})."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="auto",
        help=(
            "Where to run the model: 'auto' (default, GPU if available "
            "with enough free memory, otherwise CPU), 'cpu' or 'cuda'."
        ),
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help=(
            "Display text on screen and highlight words in real-time as "
            "they are spoken."
        ),
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--epub",
        metavar="FILE.epub",
        help="Path to an EPUB book.",
    )
    source.add_argument(
        "--text",
        dest="text_file",
        metavar="FILE.txt",
        help=(
            "Path to a UTF-8 text file, used like --epub but with a plain "
            "text: read aloud, saved with --output, or analyzed with "
            "--analyze."
        ),
    )
    parser.add_argument(
        "--project",
        nargs="?",
        const="",
        default=None,
        metavar="NAME",
        help=(
            "Create a .bts project for the EPUB or text source. If NAME is "
            "omitted, the project uses the source name."
        ),
    )
    parser.add_argument(
        "--build",
        metavar="PROJECT",
        help=(
            "Build audio files from a .bts project file; the extension is "
            "optional."
        ),
    )
    source.add_argument(
        "--clipboard",
        action="store_true",
        default=False,
        help=(
            "Read the current X11 primary selection (or clipboard) and speak "
            "it. Requires wl-paste, xclip, xsel, or xdotool to be installed."
        ),
    )
    parser.add_argument(
        "--output",
        nargs="?",
        const="",
        metavar="DIR|FILE",
        help=(
            "Save audio instead of playing it. With --epub it is the "
            "directory that receives one file per section; without a value "
            "it is named after the book. For any other text it is the name "
            "of the audio file, which is required, except with --text, "
            "where it defaults to the name of the text file. Existing "
            "audio files of a plain text are replaced."
        ),
    )
    parser.add_argument(
        "--format",
        action=_StoreExplicit,
        choices=AUDIO_FORMATS,
        default=None,
        help=(
            "Audio format of the files created with --output: 'flac' or "
            "'mp3' (variable bitrate, needs ffmpeg with libmp3lame). If "
            "omitted, it is taken from the extension of the file name, or "
            "it is 'flac'."
        ),
    )
    parser.add_argument(
        "--mp3-quality",
        type=int,
        choices=range(10),
        metavar="0-9",
        default=DEFAULT_MP3_QUALITY,
        help=(
            "Variable-bitrate quality of MP3 files: 0 is the best and "
            f"largest, 9 the smallest (default: {DEFAULT_MP3_QUALITY})."
        ),
    )
    parser.add_argument(
        "--dictionary",
        action=_StoreExplicit,
        metavar="FILE",
        help=(
            "Pronunciation dictionary: a UTF-8 text file with one "
            "'word|replacement' per line (for example 'mr|mister'). "
            "With --analyze it is, instead, the name of the file to "
            "create. "
            "The replacement is spoken instead of the word; if it is "
            "written between square brackets it is read as phonemes "
            "('mr|[mˈɪstəɹ]'). Lines starting with '#' are comments. "
            "Applies to every text that is read."
        ),
    )
    parser.add_argument(
        "--dictionary-ignore-case",
        action="store_true",
        help="Match the words of --dictionary regardless of capitalization.",
    )
    parser.add_argument(
        "--analyze",
        dest="analyze",
        action="store_true",
        help=(
            "With --epub or --text, write a file with the words that the "
            "Hunspell dictionary of --lang does not know, one 'word|word' "
            "per line, to use as a pronunciation dictionary. It is named "
            "after the source with the extension .dict (in the current "
            "directory), or as given by --dictionary. It stops if the "
            "file already exists."
        ),
    )
    ipa_action = parser.add_argument(
        "--ipa",
        nargs="?",
        const="",
        default=None,
        type=str.lower,
        metavar="LANG",
        help=(
            "With --analyze, write for each word the phonemes that Kokoro "
            "would use, as 'word|[word](/phonemes/)', so that they can be "
            "corrected by hand. They are generated with the language of "
            "--lang, or with LANG if it is given (for example, English "
            "names in a Spanish book: --lang es --ipa us)."
        ),
    )
    ipa_action.completer = ChoicesCompleter(sorted(SUPPORTED_LANGUAGES))
    parser.add_argument(
        "--sections-list",
        action="store_true",
        help="List all sections and titles in the EPUB and exit.",
    )
    parser.add_argument(
        "--section-start",
        metavar="SECTION",
        help=(
            "Start reading or exporting from the specified section index "
            "or title substring."
        ),
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="Display all supported ISO language codes and exit.",
    )
    list_voices_action = parser.add_argument(
        "--list-voices",
        nargs="?",
        const="all",
        metavar="LANG",
        help=(
            "Display available neural voices. Optionally pass an ISO code "
            "to filter."
        ),
    )
    list_voices_action.completer = ChoicesCompleter(
        sorted(SUPPORTED_LANGUAGES)
    )
    test_action = parser.add_argument(
        "--test",
        metavar="LANG",
        help=(
            "Test all available voices for the specified ISO language "
            "using preset presentation text."
        ),
    )
    test_action.completer = ChoicesCompleter(sorted(SUPPORTED_LANGUAGES))
    parser.add_argument(
        "--no-daemon",
        action="store_true",
        dest="no_daemon",
        help=(
            "Bypass the daemon and run in standalone mode even if the "
            "BTS daemon is active."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Daemon management
# ---------------------------------------------------------------------------

def _add_daemon_arguments(parser: argparse.ArgumentParser) -> None:
    """Add daemon actions to a parser."""
    sub = parser.add_subparsers(dest="action")

    start_p = sub.add_parser(
        "start", help="Start the daemon."
    )
    start_p.add_argument(
        "--fg",
        action="store_true",
        help="Run in the foreground instead of background.",
    )
    start_p.add_argument(
        "--timeout",
        type=int,
        default=30,
        metavar="MINUTES",
        help=(
            "Inactivity timeout in minutes (0 = never shutdown). "
            "Default: 30."
        ),
    )

    sub.add_parser("pause", help="Pause the current conversion.")
    sub.add_parser("stop", help="Stop the current conversion.")
    sub.add_parser(
        "next", help="Skip the current task and continue the queue."
    )
    sub.add_parser("shutdown", help="Shut down the daemon process.")
    sub.add_parser("status", help="Show daemon status.")
    sub.add_parser("list", help="List the active and pending tasks.")
    for action, help_text in (
        ("delete", "Delete a pending task."),
        ("up", "Move a pending task one position toward execution."),
        ("down", "Move a pending task one position away from execution."),
    ):
        command = sub.add_parser(action, help=help_text)
        command.add_argument("number", type=int)
    move_p = sub.add_parser("move", help="Move a pending task to a position.")
    move_p.add_argument("from_number", type=int, metavar="from")
    move_p.add_argument("to_number", type=int, metavar="to")
    sub.add_parser("play", help="Resume a paused conversion.")
    sub.add_parser("resume", help="Resume a paused conversion.")
    sub.add_parser("help", help="Show daemon command help.")


def _daemon_completion_parser() -> argparse.ArgumentParser:
    """Build the parser tree argcomplete needs for ``bts daemon``."""
    parser = argparse.ArgumentParser(prog="bts")
    commands = parser.add_subparsers(dest="command")
    daemon = commands.add_parser(
        "daemon",
        description="Manage the BTS background daemon.",
    )
    _add_daemon_arguments(daemon)
    return parser


def _is_daemon_completion() -> bool:
    """Return whether argcomplete is handling a daemon command line."""
    if not os.environ.get("_ARGCOMPLETE"):
        return False
    words = os.environ.get("COMP_LINE", "").split(maxsplit=2)
    return (
        len(words) > 1
        and os.path.basename(words[0]) in {"bts", "bts-tts"}
        and "daemon".startswith(words[1])
    )


def _handle_daemon_command(argv: list[str]) -> int:
    """Parse and execute ``bts daemon <action>`` subcommands.

    Args:
        argv: Everything after ``bts daemon`` on the command line.

    Returns:
        Process exit status.
    """
    p = argparse.ArgumentParser(
        prog="bts daemon",
        description="Manage the BTS background daemon.",
    )
    _add_daemon_arguments(p)
    parsed = p.parse_args(argv)
    if parsed.action is None:
        p.print_help()
        return 0

    if parsed.action == "help":
        p.print_help()
        return 0

    if parsed.action == "start":
        return _daemon_start(parsed.fg, parsed.timeout * 60)
    if parsed.action == "stop":
        return _daemon_stop()
    if parsed.action == "next":
        return _daemon_control("next")
    if parsed.action == "pause":
        return _daemon_control("pause")
    if parsed.action in {"play", "resume"}:
        return _daemon_control(parsed.action)
    if parsed.action == "shutdown":
        return _daemon_shutdown()
    if parsed.action == "status":
        return _daemon_status()
    if parsed.action == "list":
        return _daemon_control("list")
    if parsed.action in {"delete", "up", "down"}:
        return _daemon_control(parsed.action, number=parsed.number)
    if parsed.action == "move":
        return _daemon_control(
            "move", **{"from": parsed.from_number, "to": parsed.to_number}
        )
    return 1


def _daemon_start(foreground: bool, timeout_seconds: int) -> int:
    """Start the BTS daemon."""
    from bts.client import ping_daemon
    from bts.daemon import LOG_PATH, run_daemon

    if ping_daemon():
        print("[INFO] BTS daemon is already running.")
        return 0

    if foreground:
        print(
            "[INFO] Starting BTS daemon in foreground "
            "(press Ctrl+C to stop)…"
        )
        run_daemon(inactivity_timeout=timeout_seconds)
        return 0

    # Background mode: redirect output to a log file.
    import subprocess
    import time

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_fh = LOG_PATH.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "bts.daemon",
                f"--timeout={timeout_seconds // 60}",
            ],
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,
        )
    finally:
        log_fh.close()  # Parent does not need the file; child keeps it open.

    # Wait up to 3 s for the daemon to become responsive.
    for _ in range(30):
        time.sleep(0.1)
        if ping_daemon():
            print(
                f"[INFO] BTS daemon started (PID {proc.pid}). "
                f"Logs: {LOG_PATH}"
            )
            return 0

    print(
        f"[WARNING] Daemon process started (PID {proc.pid}) but is not "
        "responding yet — it may still be loading the model."
    )
    return 0


def _daemon_stop() -> int:
    """Stop the current conversion without shutting down the daemon."""
    return _daemon_control("stop")


def _daemon_control(action: str, **parameters: int) -> int:
    """Send a conversion control action to the daemon."""
    from bts.client import control_daemon

    result = control_daemon(action, **parameters)
    if result is not None:
        message, success = result
        prefix = "[INFO]" if success else "[ERROR]"
        print(f"{prefix} {message}")
        return 0 if success else 1
    else:
        print("[INFO] No BTS daemon is running.")
        return 1


def _daemon_shutdown() -> int:
    """Shut down the daemon process."""
    from bts.client import stop_daemon

    if stop_daemon():
        print("[INFO] BTS daemon shutting down.")
    else:
        print("[INFO] No BTS daemon is running.")
    return 0


def _daemon_status() -> int:
    """Print the status of the BTS daemon."""
    from bts.client import get_daemon_status
    from bts.daemon import PID_PATH, SOCKET_PATH

    status = get_daemon_status()
    if status:
        print(f"[INFO] Active — {status}")
        return 0

    # Check for stale files left by a crash.
    if SOCKET_PATH.exists() or PID_PATH.exists():
        print(
            "[INFO] Daemon appears to have crashed "
            "(socket or PID file exists but daemon is unresponsive)."
        )
        return 1

    print("[INFO] No BTS daemon is running.")
    return 0


# ---------------------------------------------------------------------------
# Argument preparation
# ---------------------------------------------------------------------------

def prepare_synthesis(
    args: argparse.Namespace,
    needs_player: bool,
    lang_code: str,
    needs_mp3: bool = False,
) -> str:
    """Validate the options and pick the device before loading any model.

    Args:
        args: Parsed command line arguments.
        needs_player: True if audio will be played through ``ffplay``.
        lang_code: ISO language code that will be used.
        needs_mp3: True if MP3 files will be written through ``ffmpeg``.

    Returns:
        The device to use (``"cpu"`` or ``"cuda"``).
    """
    validate_voice(lang_code, args.voice)
    if needs_player:
        require_ffplay()
    if needs_mp3:
        require_mp3_encoder()
    return resolve_device(args.device)


def load_lexicon(args: argparse.Namespace) -> PronunciationDictionary:
    """Load the pronunciation dictionary chosen on the command line.

    Args:
        args: Parsed command line arguments.

    Returns:
        The dictionary, which is empty if ``--dictionary`` was not given.

    Raises:
        TTSError: If the dictionary file cannot be read.
    """
    if not args.dictionary:
        return PronunciationDictionary({})
    lexicon = load_dictionary(args.dictionary, args.dictionary_ignore_case)
    print(
        f"[INFO] Loaded {len(lexicon)} pronunciation entries from "
        f"'{args.dictionary}'."
    )
    return lexicon


def _stdin_has_data(timeout: float = 2.0) -> bool:
    """Return True if piped input is available on standard input.

    A terminal never counts, and an open pipe that stays silent for
    ``timeout`` seconds (a background job, cron) is treated as empty
    instead of blocking forever. Use ``-`` as the text to always wait.
    """
    if sys.stdin is None or sys.stdin.isatty():
        return False
    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
    except (OSError, ValueError):
        return True  # not selectable: just read it
    return bool(ready)


def read_clipboard() -> str:
    """Return the text currently in the X11 primary selection or clipboard.

    Tries ``wl-paste``, ``xclip``, ``xsel``, and ``xdotool`` in that order.
    The primary selection is used when available because it reflects the most
    recent mouse selection; the CLIPBOARD is used as a fallback.

    Returns:
        The clipboard text, stripped of leading/trailing whitespace.

    Raises:
        TTSError: If no clipboard tool is found or the selection is empty.
    """
    import subprocess

    # tools = [
    #     ["wl-paste", "-p"],
    #     ["xclip", "-selection", "primary", "-o"],
    #     ["xclip", "-selection", "clipboard", "-o"],
    #     ["xsel", "--primary", "--output"],
    #     ["xsel", "--clipboard", "--output"],
    #     ["xdotool", "getactivewindow", "selectall", "--clearmodifiers",
    #      "type", "--clearmodifiers"],  # intentionally last / unlikely
    # ]
    # Simpler ordered list: primary first, clipboard fallback, two tools.
    ordered = [
        (["wl-paste", "-p"], "wl-paste"),
        (["xclip", "-selection", "primary", "-o"], "xclip (primary)"),
        (["xclip", "-selection", "clipboard", "-o"], "xclip (clipboard)"),
        (["xsel", "--primary", "--output"], "xsel (primary)"),
        (["xsel", "--clipboard", "--output"], "xsel (clipboard)"),
        (["xdotool", "getactivewindow"], None),   # probe only
    ]

    errors: list[str] = []
    for cmd, label in ordered:
        if label is None:
            continue  # probe entries skipped here
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0:
                text = result.stdout.strip()
                if text:
                    return text
                # Empty selection — try next tool/selection type.
                errors.append(f"{label}: empty")
            else:
                errors.append(f"{label}: exit {result.returncode}")
        except FileNotFoundError:
            errors.append(f"{label}: not found")
        except subprocess.TimeoutExpired:
            errors.append(f"{label}: timed out")

    if all("not found" in e for e in errors):
        raise TTSError(
            "No clipboard tool found. Install wl-paste, xclip or xsel:\n"
            "  sudo apt install wl-clipboard\n"
            "  sudo apt install xclip\n"
            "  sudo apt install xsel"
        )
    raise TTSError(
        "The clipboard/primary selection is empty or could not be read."
    )


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Execute the action selected on the command line.

    Args:
        args: Parsed command line arguments.
        parser: The parser, used to report usage errors.

    Returns:
        The process exit status.
    """
    if args.build is not None:
        if (
            args.project is not None
            or args.epub
            or args.text_file
            or args.clipboard
            or args.text is not None
            or args.analyze
            or args.ipa is not None
            or args.sections_list
            or args.section_start is not None
            or args.test is not None
            or args.list_languages
            or args.list_voices is not None
        ):
            parser.error(
                "--build cannot be combined with source, project, "
                "analysis, section, test, or listing options."
            )
        return build_project(
            args.build,
            output_dir=args.output,
            language_override=(
                args.lang if args.lang_explicit else None
            ),
            voice_override=(
                args.voice if args.voice_explicit else None
            ),
            format_override=(
                args.format if args.format_explicit else None
            ),
            dictionary_override=(
                args.dictionary
                if args.dictionary_explicit else None
            ),
            dictionary_ignore_case=args.dictionary_ignore_case,
            speed=args.speed,
            device=args.device,
            mp3_quality=args.mp3_quality,
            max_pause_seconds=args.max_pause,
        )

    if args.list_languages:
        show_languages()
        return 0

    if args.list_voices is not None:
        show_voices(lang_filter=args.list_voices)
        return 0

    # Infer missing extensions.
    if args.epub and not os.path.splitext(args.epub)[1]:
        args.epub += ".epub"
    if args.dictionary and not os.path.splitext(args.dictionary)[1]:
        args.dictionary += ".dict"

    if args.project is not None:
        if not (args.epub or args.text_file):
            parser.error("--project requires --epub or --text.")
        if (
            args.analyze
            or args.ipa is not None
            or args.sections_list
            or args.section_start is not None
            or args.output is not None
            or args.dictionary is not None
            or args.format is not None
            or args.voice is not None
            or args.text is not None
        ):
            parser.error(
                "--project cannot be combined with analysis, section, "
                "output, dictionary, format, or voice options."
            )
        source_file = args.epub or args.text_file
        project_path = create_project(
            source_file,
            args.project or None,
            source_type="epub" if args.epub else "text",
        )
        print(f"[INFO] Created project '{project_path}'.")
        return 0

    # Auto-detect a .dict file that matches the epub name.
    if args.epub and not args.dictionary:
        candidate = os.path.splitext(args.epub)[0] + ".dict"
        if os.path.isfile(candidate):
            args.dictionary = candidate
            print(f"[INFO] Auto-loading dictionary '{candidate}'.")

    if args.test:
        lang = args.test.lower()
        if lang not in SUPPORTED_LANGUAGES:
            raise TTSError(
                f"Unsupported language code '{args.test}' for testing. "
                "Use --list-languages to view available options."
            )
        device = prepare_synthesis(args, needs_player=True, lang_code=lang)
        return 1 if run_voice_tests(lang, device=device) else 0

    if (args.sections_list or args.section_start) and not args.epub:
        parser.error(
            "--sections-list and --section-start require --epub."
        )

    if args.ipa is not None:
        if not args.analyze:
            parser.error("--ipa requires --analyze.")
        if args.ipa and args.ipa not in SUPPORTED_LANGUAGES:
            parser.error(
                f"--ipa: unsupported language '{args.ipa}'. Use "
                "--list-languages to view the available codes."
            )

    if args.analyze:
        if not (args.epub or args.text_file):
            parser.error("--analyze requires --epub or --text.")
        if args.output is not None:
            print(
                "[INFO] --output is ignored with --analyze.",
                file=sys.stderr,
            )
        return run_analysis(args)

    if args.text and (args.epub or args.text_file):
        parser.error(
            "a text argument cannot be combined with --epub or --text."
        )

    if args.epub:
        audio_format = args.format or "flac"
        streaming = args.output is None and not args.sections_list
        exporting_mp3 = args.output is not None and audio_format == "mp3"
        device = "cpu"
        lexicon = PronunciationDictionary({})
        if not args.sections_list:
            if not args.no_daemon:
                from bts.client import try_daemon_request

                result = try_daemon_request(
                    {
                        "action": "epub",
                        "epub_path": args.epub,
                        "lang": args.lang,
                        "voice": args.voice,
                        "speed": args.speed,
                        "max_pause": args.max_pause,
                        "device": args.device,
                        "output": args.output,
                        "format": audio_format,
                        "mp3_quality": args.mp3_quality,
                        "section_start": args.section_start,
                        "dictionary": args.dictionary,
                        "dictionary_ignore_case": (
                            args.dictionary_ignore_case
                        ),
                    }
                )
                if result is not None:
                    return result
            lexicon = load_lexicon(args)
            device = prepare_synthesis(
                args,
                needs_player=streaming,
                lang_code=args.lang,
                needs_mp3=exporting_mp3,
            )
        failed = process_epub_book(
            epub_path=args.epub,
            lang_code=args.lang,
            voice=args.voice,
            output_arg=args.output,
            sections_list=args.sections_list,
            section_start=args.section_start,
            display=args.display,
            speed=args.speed,
            max_pause_seconds=args.max_pause,
            device=device,
            audio_format=audio_format,
            mp3_quality=args.mp3_quality,
            lexicon=lexicon,
        )
        return 1 if failed else 0

    default_name = None
    if args.clipboard:
        text_to_speak = read_clipboard()
    elif args.text_file:
        text_to_speak = read_text_file(args.text_file)
        default_name = os.path.splitext(os.path.basename(args.text_file))[0]
    elif args.text and args.text != "-":
        text_to_speak = args.text
    elif args.text == "-" or _stdin_has_data():
        text_to_speak = sys.stdin.read().strip()
    else:
        parser.print_help()
        return 1

    if not text_to_speak:
        raise TTSError("There is no text to synthesize (empty input).")

    if args.output is not None:
        path, audio_format = resolve_audio_output(
            args.output, args.format, default_name
        )
        if not args.no_daemon:
            from bts.client import try_daemon_request

            result = try_daemon_request(
                {
                    "action": "save",
                    "text": text_to_speak,
                    "output": path,
                    "lang": args.lang,
                    "voice": args.voice,
                    "speed": args.speed,
                    "max_pause": args.max_pause,
                    "device": args.device,
                    "format": audio_format,
                    "mp3_quality": args.mp3_quality,
                    "dictionary": args.dictionary,
                    "dictionary_ignore_case": (
                        args.dictionary_ignore_case
                    ),
                }
            )
            if result is not None:
                return result
        lexicon = load_lexicon(args)
        device = prepare_synthesis(
            args,
            needs_player=False,
            lang_code=args.lang,
            needs_mp3=audio_format == "mp3",
        )
        synthesize_text_to_file(
            prepare_text(text_to_speak, args.lang, lexicon),
            path,
            lang_code=args.lang,
            voice=args.voice,
            speed=args.speed,
            max_pause_seconds=args.max_pause,
            device=device,
            audio_format=audio_format,
            mp3_quality=args.mp3_quality,
        )
        return 0

    # Play mode.  --display requires a foreground process (karaoke), so it
    # always runs standalone even if the daemon is active.
    if not args.no_daemon and not args.display:
        from bts.client import try_daemon_request

        result = try_daemon_request(
            {
                "action": "play",
                "text": text_to_speak,
                "lang": args.lang,
                "voice": args.voice,
                "speed": args.speed,
                "max_pause": args.max_pause,
                "device": args.device,
                "dictionary": args.dictionary,
                "dictionary_ignore_case": args.dictionary_ignore_case,
            }
        )
        if result is not None:
            return result

    lexicon = load_lexicon(args)
    device = prepare_synthesis(args, needs_player=True, lang_code=args.lang)
    synthesize_and_play(
        prepare_text(text_to_speak, args.lang, lexicon),
        lang_code=args.lang,
        voice=args.voice,
        display=args.display,
        speed=args.speed,
        max_pause_seconds=args.max_pause,
        device=device,
    )
    return 0


def main(argv: list | None = None) -> int:
    """Run the command line interface.

    Args:
        argv: Command line arguments; ``sys.argv[1:]`` if omitted.

    Returns:
        The process exit status.
    """
    # Resolve the actual argument list so we can inspect it before argparse.
    _argv = list(sys.argv[1:]) if argv is None else list(argv)

    # Intercept 'bts daemon ...' before the standard parser so it does not
    # need to be a registered positional argument.
    if _argv and _argv[0] == "daemon":
        return _handle_daemon_command(_argv[1:])

    parser = (
        _daemon_completion_parser()
        if _is_daemon_completion()
        else build_parser()
    )
    argcomplete.autocomplete(parser)
    args = parser.parse_args(_argv)
    try:
        return run(args, parser)
    except GpuMemoryError as exc:
        print(f"Warning: {exc}", file=sys.stderr)
        return EXIT_GPU_BUSY
    except TTSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "\n\n[INFO] Execution aborted by user (Ctrl+C). "
            "Exiting cleanly."
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
