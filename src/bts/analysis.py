"""Spelling analysis (``--analyze``)."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import os
import re
from collections import Counter

from bts.constants import DICTIONARY_SEPARATOR, PHONEME_FORBIDDEN_RE
from bts.device import get_g2p_pipeline
from bts.epub import get_plain_text, load_sections
from bts.errors import TTSError
from bts.languages import SUPPORTED_LANGUAGES
from bts.phonemes import phoneme_markup, text_phonemes
from bts.textio import read_text_file

# A word: letters, with apostrophes or hyphens inside (don't, bien-être).
WORD_RE = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*")


# Hunspell dictionaries (<name>.aff and <name>.dic) of each language, in
# order of preference. Hunspell has none for Japanese or Chinese.
HUNSPELL_DICTIONARIES = {
    "es": ("es_ES", "es_MX", "es_AR", "es_CO", "es_US", "es"),
    "en": ("en_GB", "en_US", "en_AU", "en_CA", "en"),
    "uk": ("en_GB", "en_US", "en_AU", "en_CA", "en"),
    "us": ("en_US", "en_GB", "en_CA", "en_AU", "en"),
    "fr": ("fr_FR", "fr_CH", "fr_CA", "fr"),
    "it": ("it_IT", "it_CH", "it"),
    "pt": ("pt_BR", "pt_PT", "pt"),
}


# Where the dictionaries are searched, besides the DICPATH variable.
HUNSPELL_DIRS = (
    "/usr/share/hunspell",
    "/usr/share/myspell",
    "/usr/share/myspell/dicts",
    "/usr/local/share/hunspell",
    "~/.local/share/hunspell",
)


# Names under which the Hunspell library may be installed.
HUNSPELL_LIBRARY_NAMES = (
    "hunspell-1.7", "hunspell-1.6", "hunspell-1.5", "hunspell-1.4",
    "hunspell",
)


HUNSPELL_SONAMES = (
    "libhunspell-1.7.so.0", "libhunspell-1.6.so.0", "libhunspell-1.5.so.0",
    "libhunspell-1.4.so.0", "libhunspell.so",
)


class WordAnalyzer:
    """Find the words of a text that a language dictionary does not know.

    To support another language, subclass it and register the class with
    :func:`register_analyzer`. Languages written without spaces, such as
    Japanese or Chinese, need their own tokenizer in :meth:`words` (for
    example MeCab or jieba) and their own dictionary in :meth:`is_known`.
    """

    def words(self, text: str) -> list:
        """Split a text into the words to check, with repetitions."""
        raise NotImplementedError

    def is_known(self, word: str) -> bool:
        """Return True if the dictionary knows the word."""
        raise NotImplementedError

    def close(self) -> None:
        """Release the resources used by the dictionary."""


def find_hunspell_dictionary(lang_code: str) -> tuple:
    """Locate the Hunspell dictionary files of a language.

    The directories of the ``DICPATH`` variable are searched first, then
    ``HUNSPELL_DIRS``. Several regional variants are tried, in the order of
    ``HUNSPELL_DICTIONARIES``.

    Args:
        lang_code: ISO language code.

    Returns:
        A tuple ``(affix_path, dictionary_path)``.

    Raises:
        TTSError: If no dictionary is found.
    """
    names = HUNSPELL_DICTIONARIES.get(lang_code.lower(), ())
    directories = [
        d for d in os.environ.get("DICPATH", "").split(os.pathsep) if d
    ]
    directories += [os.path.expanduser(d) for d in HUNSPELL_DIRS]
    for name in names:
        for directory in directories:
            affix = os.path.join(directory, f"{name}.aff")
            dictionary = os.path.join(directory, f"{name}.dic")
            if os.path.isfile(affix) and os.path.isfile(dictionary):
                return affix, dictionary
    raise TTSError(
        f"No Hunspell dictionary found for '{lang_code}' (looked for "
        f"{', '.join(names)} in {', '.join(directories)}). Install the "
        "Hunspell dictionary of the language, or set DICPATH to the "
        "directory that contains it."
    )


class HunspellAnalyzer(WordAnalyzer):
    """Word analyzer that uses the Hunspell library.

    ``libhunspell`` is called directly through ``ctypes``, so no Python
    binding has to be installed or compiled.

    Args:
        lang_code: ISO language code (see ``HUNSPELL_DICTIONARIES``).

    Raises:
        TTSError: If the library or the dictionary is not found.
    """

    SEPARATORS = re.compile(r"['’\-]")

    def __init__(self, lang_code: str) -> None:
        self._handle = None
        self._lib = self._load_library()
        affix, dictionary = find_hunspell_dictionary(lang_code)
        self._handle = self._lib.Hunspell_create(
            os.fsencode(affix), os.fsencode(dictionary)
        )
        if not self._handle:
            raise TTSError(f"Hunspell cannot load '{dictionary}'.")
        encoding = self._lib.Hunspell_get_dic_encoding(self._handle)
        self._encoding = (encoding or b"utf-8").decode("ascii", "replace")
        try:
            "".encode(self._encoding)
        except LookupError:
            self._encoding = "utf-8"

    @staticmethod
    def _load_library():
        """Load ``libhunspell`` and declare the functions that are used."""
        candidates = [
            ctypes.util.find_library(name)
            for name in HUNSPELL_LIBRARY_NAMES
        ]
        candidates += HUNSPELL_SONAMES
        for name in candidates:
            if not name:
                continue
            try:
                lib = ctypes.CDLL(name)
            except OSError:
                continue
            if not hasattr(lib, "Hunspell_create"):
                continue
            lib.Hunspell_create.restype = ctypes.c_void_p
            lib.Hunspell_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
            lib.Hunspell_destroy.restype = None
            lib.Hunspell_destroy.argtypes = [ctypes.c_void_p]
            lib.Hunspell_spell.restype = ctypes.c_int
            lib.Hunspell_spell.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            lib.Hunspell_get_dic_encoding.restype = ctypes.c_char_p
            lib.Hunspell_get_dic_encoding.argtypes = [ctypes.c_void_p]
            return lib
        raise TTSError(
            "The Hunspell library (libhunspell) was not found. Install it "
            "together with the Hunspell dictionary of the language."
        )

    def words(self, text: str) -> list:
        """Return the words of a text; single letters are ignored."""
        return [word for word in WORD_RE.findall(text) if len(word) > 1]

    def is_known(self, word: str) -> bool:
        """Return True if Hunspell accepts the word.

        If the whole word is rejected but it contains apostrophes or
        hyphens, it is accepted when every part is known (single-letter
        parts, as in ``l'homme``, are accepted).
        """
        word = word.replace("’", "'")
        if self._spell(word):
            return True
        parts = [part for part in self.SEPARATORS.split(word) if part]
        if len(parts) < 2:
            return False
        return all(len(part) == 1 or self._spell(part) for part in parts)

    def close(self) -> None:
        """Free the Hunspell dictionary."""
        if self._handle is not None:
            self._lib.Hunspell_destroy(self._handle)
            self._handle = None

    def _spell(self, word: str) -> bool:
        """Ask Hunspell about a single word."""
        try:
            data = word.encode(self._encoding)
        except UnicodeEncodeError:
            return False  # characters the dictionary cannot even represent
        return bool(self._lib.Hunspell_spell(self._handle, data))


ANALYZER_FACTORIES: dict = {}


def register_analyzer(lang_code: str, factory) -> None:
    """Register the word analyzer of a language.

    This is the extension point for the languages Hunspell does not
    cover. For example, for Japanese::

        class JapaneseAnalyzer(WordAnalyzer):
            def words(self, text): ...      # tokenize with MeCab
            def is_known(self, word): ...   # look it up in a dictionary

        register_analyzer("ja", lambda lang_code: JapaneseAnalyzer())

    Args:
        lang_code: ISO language code.
        factory: Callable that receives the language code and returns a
            :class:`WordAnalyzer`.
    """
    ANALYZER_FACTORIES[lang_code.lower()] = factory


for _code in HUNSPELL_DICTIONARIES:
    register_analyzer(_code, HunspellAnalyzer)


def get_analyzer(lang_code: str) -> WordAnalyzer:
    """Create the word analyzer of a language.

    Raises:
        TTSError: If the language has no analyzer or it cannot be created.
    """
    factory = ANALYZER_FACTORIES.get(lang_code.lower())
    if factory is None:
        raise TTSError(
            f"Spelling analysis is not available for '{lang_code}': "
            "Hunspell has no dictionaries for it. Support can be added "
            "with register_analyzer()."
        )
    return factory(lang_code)


def analyze_text(text: str, analyzer: WordAnalyzer) -> list:
    """List the words of a text that the dictionary does not know.

    Words that differ only in capitalization are listed once, with the
    spelling that appears most often (the first one if there is a tie).

    Args:
        text: Text to analyze.
        analyzer: Analyzer of the language of the text.

    Returns:
        The unknown words, sorted alphabetically.
    """
    spellings: dict = {}
    for word, number in Counter(analyzer.words(text)).items():
        if not analyzer.is_known(word):
            spellings.setdefault(word.casefold(), {})[word] = number
    unknown = [max(group, key=group.get) for group in spellings.values()]
    return sorted(unknown, key=lambda word: (word.casefold(), word))


def default_dict_path(source_path: str) -> str:
    """Name of the analysis file: the one of the source, with ``.dict``.

    The file is created in the current directory, like the folder of an
    EPUB export.
    """
    stem = os.path.splitext(os.path.basename(source_path))[0]
    return f"{stem}.dict"


def add_phonemes(words: list, lang_code: str) -> tuple:
    """Pair every word with the phonemes that Kokoro would use for it.

    These are the phonemes of the G2P of Kokoro itself, the same ones used
    when the word is read, so they are always valid for the voice. A word
    whose phonemes cannot be obtained keeps the word as its replacement.

    Args:
        words: The words to look up.
        lang_code: ISO language code.

    Returns:
        A tuple ``(entries, missing)``. ``entries`` is a list of
        ``(word, replacement)`` pairs, where the replacement is
        ``[word](/phonemes/)``; ``missing`` counts the words without
        phonemes.
    """
    pipeline = get_g2p_pipeline(SUPPORTED_LANGUAGES[lang_code]["internal"])
    entries = []
    missing = 0
    for word in words:
        try:
            phonemes = text_phonemes(pipeline, word)
        except Exception:
            phonemes = ""
        if phonemes and not PHONEME_FORBIDDEN_RE.search(phonemes):
            entries.append((word, phoneme_markup(word, phonemes)))
        else:
            entries.append((word, word))
            missing += 1
    return entries, missing


def write_word_list(path: str, entries: list) -> None:
    """Write the entries, one ``word|replacement`` per line.

    The file is created exclusively: an existing file is never replaced.

    Args:
        path: File to create.
        entries: List of ``(word, replacement)`` pairs.

    Raises:
        TTSError: If the file exists or cannot be written.
    """
    try:
        with open(path, "x", encoding="utf-8", newline="\n") as handle:
            for word, replacement in entries:
                handle.write(f"{word}{DICTIONARY_SEPARATOR}{replacement}\n")
    except FileExistsError as exc:
        raise TTSError(f"The file '{path}' already exists.") from exc
    except OSError as exc:
        raise TTSError(f"Cannot write '{path}': {exc}") from exc


def run_analysis(args: argparse.Namespace) -> int:
    """Create a pronunciation dictionary with the unknown words of a text.

    The text comes from ``--epub`` or ``--text``. The result is named after
    the source (``book.epub`` gives ``book.dict``) unless ``--dictionary``
    gives another name. Each line is ``word|word``, or with ``--ipa``
    ``word|[word](/phonemes/)``, so that it can be edited and used later as
    a pronunciation dictionary.

    When ``--ipa`` is given without a language code the phonemes are
    generated with the same language as ``--lang``. When ``--ipa LANG``
    provides a different code (e.g. ``--lang es --ipa us`` for English
    names inside a Spanish book), *LANG* is used for the IPA lookup while
    ``--lang`` still governs the spelling analysis.

    Args:
        args: Parsed command line arguments.

    Returns:
        The process exit status.

    Raises:
        TTSError: If the result file already exists (checked before doing
            anything else), the language has no dictionary, or the source
            cannot be read.
    """
    source = args.epub or args.text_file
    target = args.dictionary or default_dict_path(source)
    if os.path.exists(target):
        raise TTSError(
            f"The analysis file '{target}' already exists. Delete it or "
            "choose another name with --dictionary."
        )

    analyzer = get_analyzer(args.lang)
    try:
        if args.epub:
            sections = load_sections(args.epub)
            text = "\n".join(get_plain_text(sec.soup) for sec in sections)
        else:
            text = read_text_file(args.text_file)
        print(f"[ANALYSIS] Checking '{source}' (language '{args.lang}')...")
        words = analyze_text(text, analyzer)
    finally:
        analyzer.close()

    if args.ipa is not None and args.ipa is not False:
        # ``args.ipa`` is:
        #   - a non-empty string  → language override supplied by the user
        #   - an empty string     → ``--ipa`` without an argument; use --lang
        ipa_lang = args.ipa if args.ipa else args.lang
        hint = (
            f" (IPA language: '{ipa_lang}')"
            if ipa_lang != args.lang
            else ""
        )
        print(
            f"[ANALYSIS] Looking up the phonemes of {len(words)} words"
            + hint + "..."
        )
        entries, missing = add_phonemes(words, ipa_lang)
    else:
        entries, missing = [(word, word) for word in words], 0

    write_word_list(target, entries)
    print(f"[ANALYSIS] {len(words)} unknown words written to '{target}'.")
    if missing:
        print(
            f"[ANALYSIS] No phonemes for {missing} words; they are left "
            f"as 'word{DICTIONARY_SEPARATOR}word'."
        )
    return 0
