"""Kokoro-82M local neural text-to-speech (TTS) client.

Generates speech locally with the Kokoro-82M model and either streams it to
``ffplay`` or exports the sections of an EPUB book as FLAC files.

Features:
    * Text from the command line, from standard input or from a text file
      (``--txt``), read aloud or saved as an audio file (``--output``).
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
    * Runs on the CPU by default. The GPU is optional (``--device``); when it
      is busy or full, a clear message is shown instead of a crash.

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
    bts --txt story.txt --output story.mp3
    bts "Hola" --output hola.flac
    bts --epub book.epub --analyze
    bts --epub book.epub --analyze --ipa
    bts --epub book.epub --analyze --lang es --ipa us


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

from bts.analysis import run_analysis
from bts.audio import require_ffplay, require_mp3_encoder
from bts.constants import AUDIO_FORMATS, DEFAULT_MP3_QUALITY, EXIT_GPU_BUSY
from bts.device import resolve_device
from bts.dictionary import PronunciationDictionary, load_dictionary
from bts.epub import process_epub_book
from bts.errors import GpuMemoryError, TTSError
from bts.languages import (
    SUPPORTED_LANGUAGES,
    show_languages,
    show_voices,
    validate_voice,
)
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
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        default="es",
        type=str.lower,
        choices=sorted(SUPPORTED_LANGUAGES),
        help="Language code. Default: 'es'.",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help=(
            "Neural voice ID to use. If omitted, uses the first verified "
            "voice for the selected language."
        ),
    )
    parser.add_argument(
        "--speed",
        type=positive_float,
        default=1.0,
        help="Speech speed multiplier (default: 1.0).",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help=(
            "Where to run the model: 'cpu' (default), 'cuda' (GPU) or "
            "'auto' (GPU if it has free memory, otherwise CPU)."
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
        "--txt",
        metavar="FILE.txt",
        help=(
            "Path to a UTF-8 text file, used like --epub but with a plain "
            "text: read aloud, saved with --output, or analyzed with "
            "--analyze."
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
            "of the audio file, which is required, except with --txt, "
            "where it defaults to the name of the text file. Existing "
            "audio files of a plain text are replaced."
        ),
    )
    parser.add_argument(
        "--format",
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
            "With --epub or --txt, write a file with the words that the "
            "Hunspell dictionary of --lang does not know, one 'word|word' "
            "per line, to use as a pronunciation dictionary. It is named "
            "after the source with the extension .dict (in the current "
            "directory), or as given by --dictionary. It stops if the "
            "file already exists."
        ),
    )
    parser.add_argument(
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
    parser.add_argument(
        "--section-list",
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
    parser.add_argument(
        "--list-voices",
        nargs="?",
        const="all",
        metavar="LANG",
        help=(
            "Display available neural voices. Optionally pass an ISO code "
            "to filter."
        ),
    )
    parser.add_argument(
        "--test",
        metavar="LANG",
        help=(
            "Test all available voices for the specified ISO language "
            "using preset presentation text."
        ),
    )
    return parser


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


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Execute the action selected on the command line.

    Args:
        args: Parsed command line arguments.
        parser: The parser, used to report usage errors.

    Returns:
        The process exit status.
    """
    if args.list_languages:
        show_languages()
        return 0

    if args.list_voices is not None:
        show_voices(lang_filter=args.list_voices)
        return 0

    if args.test:
        lang = args.test.lower()
        if lang not in SUPPORTED_LANGUAGES:
            raise TTSError(
                f"Unsupported language code '{args.test}' for testing. "
                "Use --list-languages to view available options."
            )
        device = prepare_synthesis(args, needs_player=True, lang_code=lang)
        return 1 if run_voice_tests(lang, device=device) else 0

    if (args.section_list or args.section_start) and not args.epub:
        parser.error(
            "--section-list and --section-start require --epub."
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
        if not (args.epub or args.txt):
            parser.error("--analyze requires --epub or --txt.")
        if args.output is not None:
            print(
                "[INFO] --output is ignored with --analyze.",
                file=sys.stderr,
            )
        return run_analysis(args)

    if args.text and (args.epub or args.txt):
        parser.error(
            "a text argument cannot be combined with --epub or --txt."
        )

    if args.epub:
        audio_format = args.format or "flac"
        streaming = args.output is None and not args.section_list
        exporting_mp3 = args.output is not None and audio_format == "mp3"
        device = "cpu"
        lexicon = PronunciationDictionary({})
        if not args.section_list:
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
            section_list=args.section_list,
            section_start=args.section_start,
            display=args.display,
            speed=args.speed,
            device=device,
            audio_format=audio_format,
            mp3_quality=args.mp3_quality,
            lexicon=lexicon,
        )
        return 1 if failed else 0

    default_name = None
    if args.txt:
        text_to_speak = read_text_file(args.txt)
        default_name = os.path.splitext(os.path.basename(args.txt))[0]
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
        lexicon = load_lexicon(args)
        device = prepare_synthesis(
            args,
            needs_player=False,
            lang_code=args.lang,
            needs_mp3=audio_format == "mp3",
        )
        synthesize_text_to_file(
            lexicon.apply(text_to_speak),
            path,
            lang_code=args.lang,
            voice=args.voice,
            speed=args.speed,
            device=device,
            audio_format=audio_format,
            mp3_quality=args.mp3_quality,
        )
        return 0

    lexicon = load_lexicon(args)
    device = prepare_synthesis(args, needs_player=True, lang_code=args.lang)
    synthesize_and_play(
        lexicon.apply(text_to_speak),
        lang_code=args.lang,
        voice=args.voice,
        display=args.display,
        speed=args.speed,
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
    parser = build_parser()
    args = parser.parse_args(argv)
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
