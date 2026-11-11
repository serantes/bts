"""Pronunciation dictionary."""

from __future__ import annotations

import re
import sys

from bts.constants import DICTIONARY_SEPARATOR
from bts.errors import TTSError
from bts.phonemes import parse_phonemes, phoneme_markup


class PronunciationDictionary:
    """Replace words in a text so that they are pronounced as wanted.

    Words are matched as whole words: ``mr`` does not match inside ``mrs``.
    If a word starts or ends with punctuation (for example ``Mr.``), no word
    boundary is required on that side. When several entries could match at
    the same place, the longest one wins.

    A replacement written between square brackets, such as ``[mˈɪstəɹ]``,
    is read as phonemes: it is turned into Kokoro's inline syntax
    ``[mr](/mˈɪstəɹ/)``. Kokoro uses the phoneme set of its Misaki G2P, which
    is close to the IPA.

    Args:
        entries: Mapping from the text found in the book to the text that
            should be spoken instead.
        ignore_case: If True, ``mr`` also matches ``Mr`` and ``MR``.

    Raises:
        ValueError: If a phoneme replacement contains any of ``[ ] ( ) /``.
    """

    def __init__(self, entries: dict, ignore_case: bool = False) -> None:
        entries = {
            word: self._to_spoken(word, spoken)
            for word, spoken in entries.items()
            if word
        }
        self._ignore_case = ignore_case
        self._replacements = {
            self._key(word): spoken for word, spoken in entries.items()
        }
        alternatives = [
            self._word_regex(word)
            for word in sorted(entries, key=len, reverse=True)
        ]
        flags = re.IGNORECASE if ignore_case else 0
        self._pattern = (
            re.compile("|".join(alternatives), flags)
            if alternatives
            else None
        )

    def __len__(self) -> int:
        return len(self._replacements)

    def apply(self, text: str) -> str:
        """Return ``text`` with every dictionary word replaced.

        Args:
            text: Text that will be synthesized.

        Returns:
            The text with the replacements applied.
        """
        if self._pattern is None:
            return text
        return self._pattern.sub(self._replace, text)

    @staticmethod
    def _to_spoken(word: str, spoken: str) -> str:
        """Return the text that is inserted in place of ``word``.

        The label of the inline syntax is made of the letters and digits of
        the word, because Kokoro expects a single token there.
        """
        phonemes = parse_phonemes(spoken)
        if phonemes is None:
            return spoken
        if not phonemes:
            return ""
        return phoneme_markup(word, phonemes)

    def _key(self, word: str) -> str:
        """Return the lookup key of a word."""
        return word.casefold() if self._ignore_case else word

    def _replace(self, match: re.Match) -> str:
        """Return the replacement for a regular expression match."""
        word = match.group(0)
        return self._replacements.get(self._key(word), word)

    @staticmethod
    def _word_regex(word: str) -> str:
        """Build the regular expression that matches one dictionary word."""
        start = r"(?<!\w)" if re.match(r"\w", word[0]) else ""
        end = r"(?!\w)" if re.match(r"\w", word[-1]) else ""
        return f"{start}{re.escape(word)}{end}"


def load_dictionary(
    path: str, ignore_case: bool = False
) -> PronunciationDictionary:
    """Read a pronunciation dictionary from a ``|``-separated text file.

    Each line holds the text found in the book, a ``|`` character, and the
    text to speak instead, for example ``mr|mister``. The file must be
    UTF-8 and spaces around both columns are ignored. Blank lines and lines
    starting with ``#`` are ignored, columns after the second are ignored,
    and an empty replacement removes the word. A replacement between square
    brackets (``mr|[mˈɪstəɹ]``) is read as phonemes. Malformed lines are
    reported and skipped. If a word appears more than once, the last entry
    wins.

    Args:
        path: Path to the dictionary file.
        ignore_case: Match words regardless of their capitalization.

    Returns:
        The loaded dictionary.

    Raises:
        TTSError: If the file cannot be read.
    """
    entries: dict = {}
    try:
        with open(path, encoding="utf-8-sig") as handle:
            for number, raw_line in enumerate(handle, 1):
                line = raw_line.rstrip("\r\n")
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                columns = line.split(DICTIONARY_SEPARATOR)
                word = columns[0].strip()
                if len(columns) < 2 or not word:
                    print(
                        f"Warning: {path}:{number}: expected "
                        f"'word{DICTIONARY_SEPARATOR}replacement'; "
                        "line skipped.",
                        file=sys.stderr,
                    )
                    continue
                spoken = columns[1].strip()
                try:
                    parse_phonemes(spoken)
                except ValueError as exc:
                    print(
                        f"Warning: {path}:{number}: {exc}; line skipped.",
                        file=sys.stderr,
                    )
                    continue
                entries[word] = spoken
    except OSError as exc:
        raise TTSError(f"Cannot read dictionary '{path}': {exc}") from exc
    except UnicodeDecodeError as exc:
        raise TTSError(
            f"Dictionary '{path}' is not valid UTF-8: {exc}"
        ) from exc
    return PronunciationDictionary(entries, ignore_case=ignore_case)
