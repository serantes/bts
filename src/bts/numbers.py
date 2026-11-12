"""Number-to-words normalisation for all supported BTS languages.

Before synthesis, numeric tokens in the input text are replaced by their
word equivalents so that the TTS model pronounces them naturally.  For
example, in Spanish ``"El capítulo 3 cuesta 1.500€"`` becomes
``"El capítulo tres cuesta mil quinientos€"``.

Supported numeric forms
-----------------------
* **Integers**: ``42``, ``-7``, ``1000``
* **Decimals**: ``3.14``, ``2,50`` (comma treated as decimal separator when
  the fractional part is not exactly three digits)
* **Thousands-separated integers**: ``1.500``, ``12.000.000`` (dot used as
  thousands separator in many European locales — detected when every group
  after a dot has exactly three digits)
* **Ordinal suffixes**:
  - English: ``1st``, ``2nd``, ``3rd``, ``4th``
  - Spanish/Portuguese/Italian: ``1.º`` / ``1.ª``, ``2.°``
  - French: ``1er``, ``2e``

Numbers that appear inside version strings (``v1.2.3``), IP addresses
(``192.168.0.1``), Kokoro phoneme markup (``[word](/phonemes/)``), or
immediately after a letter are left untouched.

Language mapping
----------------
BTS ISO codes map to ``num2words`` language codes as follows:

+---------+---------------------------+----------------------------+
| BTS ISO | Description               | num2words lang             |
+---------+---------------------------+----------------------------+
| ``es``  | Spanish                   | ``es``                     |
| ``en``  | English (generic)         | ``en``                     |
| ``uk``  | English – British         | ``en``  (not Ukrainian!)   |
| ``us``  | English – American        | ``en``                     |
| ``fr``  | French                    | ``fr``                     |
| ``it``  | Italian                   | ``it``                     |
| ``ja``  | Japanese                  | ``ja``                     |
| ``zh``  | Mandarin Chinese          | *no support – skipped*     |
| ``pt``  | Portuguese                | ``pt``                     |
+---------+---------------------------+----------------------------+

Chinese (``zh``) is not supported by ``num2words``; digits in Chinese text
are left as-is (the Kokoro G2P handles them reasonably).
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from bts.constants import IGNORED_STRINGS

# ---------------------------------------------------------------------------
# Language mapping: BTS ISO code → num2words lang code, or None to skip.
# ---------------------------------------------------------------------------

#: BTS ISO code → ``num2words`` language code.  ``None`` means the language
#: is not supported by num2words and numbers are left unchanged.
_LANG_MAP: dict[str, str | None] = {
    "es": "es",
    "en": "en",
    "uk": "en",   # BTS "uk" = British English, NOT Ukrainian
    "us": "en",
    "fr": "fr",
    "it": "it",
    "ja": "ja",
    "zh": None,   # num2words has no Mandarin Chinese support
    "pt": "pt",
}

# ---------------------------------------------------------------------------
# Ordinal-suffix patterns per language family
# ---------------------------------------------------------------------------

# English: 1st, 2nd, 3rd, 4th, …
_ORDINAL_EN = re.compile(
    r"(?<![a-zA-Z0-9_/])(-?\d+)(st|nd|rd|th)(?![a-zA-Z0-9_/])", re.IGNORECASE
)

# Spanish / Portuguese / Italian: 1.º 1.ª 2.° (with optional dot before sign)
_ORDINAL_ES = re.compile(r"(?<![a-zA-Z0-9_/])(-?\d+)\.?[º°ª](?![a-zA-Z0-9_/])")

# French: 1er, 1re, 2e, 3e, … (must follow immediately)
_ORDINAL_FR = re.compile(
    r"(?<![a-zA-Z0-9_/])(-?\d+)(er|re|e)(?![a-zA-Z0-9_/])", re.IGNORECASE
)

# Italian ordinals: same suffix notation as Spanish; also handles plain
# ordinals because Italian often writes «3°».  Reuse _ORDINAL_ES.

# ---------------------------------------------------------------------------
# General number pattern
# ---------------------------------------------------------------------------

# Matches integers and decimals (with . or , as decimal/thousands separator).
# Uses ASCII-only word boundaries ([a-zA-Z0-9_.]) instead of \w so that:
#   - CJK characters adjacent to digits are treated as non-word-boundaries
#     (Japanese "3個" → digit IS replaced).
#   - Digits that are part of version strings ("v1.2.3") are NOT replaced,
#     because the leading dot counts as a continuation character.
#
# Groups:
#   1 → optional leading minus
#   2 → digit groups separated by . or ,
_NUMBER_RE = re.compile(
    r"(?<![/a-zA-Z0-9_.])(-?)(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?"
    r"|\d+(?:[.,]\d+)?)(?![/a-zA-Z0-9_])"
)

# Kokoro phoneme markup: [word](/phonemes/) — must not be touched.
_PHONEME_MARKUP_RE = re.compile(r"\[.*?\]\(/[^)]*?/\)")


# ---------------------------------------------------------------------------
# Decimal / integer parsing helpers
# ---------------------------------------------------------------------------

def _parse_number(raw: str, sign: str) -> tuple[Decimal | int, bool]:
    """Return ``(value, is_ordinal=False)`` for a raw digit string.

    Heuristic for the ambiguous European decimal / thousands separator:

    * If every group after a separator has **exactly** 3 digits and there is
      more than one group → thousands separator (integer).
    * Otherwise → decimal separator.

    The ``sign`` argument is ``"-"`` or ``""``.
    """
    # Normalise: replace comma with dot for Decimal parsing
    normalised = raw.replace(",", ".")

    # Detect thousands-separator pattern: 1.500 / 1.500.000 / 12,000,000
    parts = re.split(r"[.,]", raw)
    if (
        len(parts) > 1
        and all(len(p) == 3 for p in parts[1:])
    ):
        # All fractional groups have exactly 3 digits → thousands separator
        integer_str = "".join(parts)
        return int(sign + integer_str), False

    # Regular decimal or integer
    try:
        value = Decimal(sign + normalised)
    except InvalidOperation:
        raise ValueError(f"Cannot parse '{sign}{raw}' as a number")

    if value == value.to_integral_value():
        return int(value), False
    return value, False


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def _convert_number(
    value: int | Decimal,
    n2w_lang: str,
    to: str = "cardinal",
) -> str:
    """Call ``num2words`` and return the result, or raise ``ValueError``."""
    from num2words import num2words  # lazy import – heavy on first call

    try:
        return num2words(value, lang=n2w_lang, to=to)
    except (NotImplementedError, TypeError, OverflowError) as exc:
        raise ValueError(f"num2words cannot convert {value!r}: {exc}") from exc


def _replace_ordinals_en(text: str, n2w_lang: str) -> str:
    """Replace English ordinal suffixes (1st, 2nd …)."""
    def _sub(m: re.Match) -> str:
        try:
            value = int(m.group(1))
            return _convert_number(value, n2w_lang, to="ordinal")
        except (ValueError, OverflowError):
            return m.group(0)

    return _ORDINAL_EN.sub(_sub, text)


def _replace_ordinals_es(text: str, n2w_lang: str) -> str:
    """Replace Spanish/Portuguese/Italian ordinal notation (1.º, 2.ª …)."""
    def _sub(m: re.Match) -> str:
        try:
            value = int(m.group(1))
            return _convert_number(value, n2w_lang, to="ordinal")
        except (ValueError, OverflowError):
            return m.group(0)

    return _ORDINAL_ES.sub(_sub, text)


def _replace_ordinals_fr(text: str, n2w_lang: str) -> str:
    """Replace French ordinal suffixes (1er, 2e …)."""
    def _sub(m: re.Match) -> str:
        try:
            value = int(m.group(1))
            return _convert_number(value, n2w_lang, to="ordinal")
        except (ValueError, OverflowError):
            return m.group(0)

    return _ORDINAL_FR.sub(_sub, text)


def _replace_cardinals(text: str, n2w_lang: str) -> str:
    """Replace all remaining numeric tokens with cardinal words."""
    def _sub(m: re.Match) -> str:
        sign, raw = m.group(1), m.group(2)
        try:
            value, _ = _parse_number(raw, sign)
            return _convert_number(value, n2w_lang, to="cardinal")
        except (ValueError, OverflowError):
            return m.group(0)

    return _NUMBER_RE.sub(_sub, text)


# ---------------------------------------------------------------------------
# Phoneme-markup protection
# ---------------------------------------------------------------------------

def _protect_markup(text: str) -> tuple[str, dict]:
    """Replace Kokoro phoneme markup spans with placeholders.

    Returns the modified text and a dict mapping placeholder → original span,
    so the caller can restore them afterwards.
    """
    placeholders: dict[str, str] = {}
    counter = [0]

    def _replace(m: re.Match) -> str:
        key = f"\x00MARKUP{counter[0]}\x00"
        counter[0] += 1
        placeholders[key] = m.group(0)
        return key

    return _PHONEME_MARKUP_RE.sub(_replace, text), placeholders


def _restore_markup(text: str, placeholders: dict) -> str:
    for key, original in placeholders.items():
        text = text.replace(key, original)
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_numbers(text: str, lang_code: str) -> str:
    """Replace numeric tokens in *text* with their word equivalents.

    The conversion uses the ``num2words`` library with the appropriate
    language for *lang_code*.  If the language is not supported by
    ``num2words`` (currently ``zh``), the text is returned unchanged.

    Numbers inside Kokoro phoneme markup (``[word](/phonemes/)``) are
    protected and never replaced.

    Args:
        text: Text to normalise; may contain phoneme markup.
        lang_code: BTS ISO language code (``"es"``, ``"en"``, ``"us"`` …).

    Returns:
        Text with numeric tokens replaced by words in the target language.
    """
    code = lang_code.lower()
    n2w_lang = _LANG_MAP.get(code)
    if n2w_lang is None:
        return text  # Language not supported (e.g. zh); leave as-is.

    # Protect Kokoro phoneme markup spans from substitution.
    text, placeholders = _protect_markup(text)

    # 1. Ordinal suffixes (must come before cardinal replacement).
    if code in ("en", "uk", "us"):
        text = _replace_ordinals_en(text, n2w_lang)
    elif code in ("es", "pt", "it"):
        text = _replace_ordinals_es(text, n2w_lang)
    elif code == "fr":
        text = _replace_ordinals_fr(text, n2w_lang)

    # 2. Cardinals (integers and decimals).
    text = _replace_cardinals(text, n2w_lang)

    # Restore protected spans.
    return _restore_markup(text, placeholders)


def prepare_text(
    text: str,
    lang_code: str,
    lexicon,
) -> str:
    """Normalise numbers and apply the pronunciation dictionary.

    This is the single entry point that callers should use to prepare text
    for synthesis.  It combines :func:`normalize_numbers` (number-to-words)
    with :meth:`~bts.dictionary.PronunciationDictionary.apply` (custom
    pronunciation entries) in the correct order: numbers are expanded first
    so that dictionary entries can match the resulting words if needed.

    Args:
        text: Raw text to prepare.
        lang_code: BTS ISO language code (``"es"``, ``"en"`` …).
        lexicon: A :class:`~bts.dictionary.PronunciationDictionary` instance.

    Returns:
        Text ready for synthesis.
    """
    return lexicon.apply(
        normalize_numbers(remove_ignored_strings(text), lang_code)
    )


def remove_ignored_strings(text: str) -> str:
    """Remove configured strings that must be treated as absent."""
    for ignored in IGNORED_STRINGS:
        text = text.replace(ignored, "")
    return text
