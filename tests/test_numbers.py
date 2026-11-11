"""Tests for the number-to-words normalisation module."""

from __future__ import annotations

import pytest

from bts.numbers import normalize_numbers, prepare_text
from bts.dictionary import PronunciationDictionary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_lexicon():
    return PronunciationDictionary({})


# ---------------------------------------------------------------------------
# Basic cardinal conversion
# ---------------------------------------------------------------------------

class TestCardinals:
    def test_integer_es(self):
        assert normalize_numbers("hay 3 manzanas", "es") == "hay tres manzanas"

    def test_integer_en(self):
        assert normalize_numbers("I have 5 apples", "en") == "I have five apples"

    def test_integer_us(self):
        # BTS "us" → num2words "en"
        assert normalize_numbers("page 42", "us") == "page forty-two"

    def test_integer_uk(self):
        # BTS "uk" = British English, NOT Ukrainian
        assert normalize_numbers("chapter 7", "uk") == "chapter seven"

    def test_integer_fr(self):
        assert normalize_numbers("j'ai 10 chats", "fr") == "j'ai dix chats"

    def test_integer_it(self):
        assert normalize_numbers("ho 3 gatti", "it") == "ho tre gatti"

    def test_integer_pt(self):
        assert normalize_numbers("tenho 3 gatos", "pt") == "tenho três gatos"

    def test_integer_ja(self):
        # Japanese: num2words returns kanji numerals
        result = normalize_numbers("3個", "ja")
        assert "3" not in result  # digit replaced

    def test_zh_unchanged(self):
        # Mandarin Chinese is not supported — text must be untouched
        text = "有3只猫"
        assert normalize_numbers(text, "zh") == text

    def test_negative_number(self):
        assert normalize_numbers("temperatura de -5 grados", "es") == \
               "temperatura de menos cinco grados"

    def test_zero(self):
        assert normalize_numbers("hay 0 errores", "es") == "hay cero errores"

    def test_large_number(self):
        result = normalize_numbers("1000000 personas", "es")
        assert "1" not in result
        assert "1000000" not in result

    def test_multiple_numbers_in_sentence(self):
        result = normalize_numbers("entre 3 y 7 manzanas", "es")
        assert "3" not in result
        assert "7" not in result
        assert "tres" in result
        assert "siete" in result


# ---------------------------------------------------------------------------
# Thousands separator (European notation: 1.500 = 1500)
# ---------------------------------------------------------------------------

class TestThousandsSeparator:
    def test_dot_thousands_es(self):
        # 1.500 in Spanish → "mil quinientos"
        result = normalize_numbers("cuesta 1.500 euros", "es")
        assert "1.500" not in result
        assert "quinientos" in result.lower() or "mil" in result.lower()

    def test_dot_thousands_multiple_groups(self):
        # 1.000.000 → one million
        result = normalize_numbers("tiene 1.000.000 habitantes", "es")
        assert "1.000.000" not in result
        assert "millón" in result.lower() or "millon" in result.lower()

    def test_comma_thousands_en(self):
        # 1,000 in English → "one thousand"
        result = normalize_numbers("costs 1,000 dollars", "en")
        assert "1,000" not in result
        assert "thousand" in result.lower()


# ---------------------------------------------------------------------------
# Decimal numbers
# ---------------------------------------------------------------------------

class TestDecimals:
    def test_dot_decimal_es(self):
        # 3.14 → "tres punto uno cuatro" (num2words digit-by-digit for es)
        result = normalize_numbers("pi es 3.14", "es")
        assert "3.14" not in result

    def test_comma_decimal_es(self):
        # 2,50 → decimal (fractional part is 2 digits, not 3)
        result = normalize_numbers("cuesta 2,50 euros", "es")
        assert "2,50" not in result

    def test_decimal_en(self):
        result = normalize_numbers("it costs 1.5 dollars", "en")
        assert "1.5" not in result


# ---------------------------------------------------------------------------
# Ordinal suffixes
# ---------------------------------------------------------------------------

class TestOrdinals:
    def test_english_1st(self):
        result = normalize_numbers("the 1st prize", "en")
        assert "1st" not in result
        assert "first" in result.lower()

    def test_english_2nd(self):
        result = normalize_numbers("the 2nd floor", "us")
        assert "2nd" not in result
        assert "second" in result.lower()

    def test_english_3rd(self):
        result = normalize_numbers("the 3rd time", "uk")
        assert "3rd" not in result
        assert "third" in result.lower()

    def test_english_4th(self):
        result = normalize_numbers("4th of July", "en")
        assert "4th" not in result
        assert "fourth" in result.lower()

    def test_spanish_ordinal_masculine(self):
        # 2.º → "segundo"
        result = normalize_numbers("el 2.º capítulo", "es")
        assert "2" not in result
        assert "segundo" in result.lower()

    def test_spanish_ordinal_feminine(self):
        # 1.ª → "primera"
        result = normalize_numbers("la 1.ª edición", "es")
        assert "1" not in result

    def test_french_ordinal_1er(self):
        result = normalize_numbers("le 1er étage", "fr")
        assert "1er" not in result
        assert "premier" in result.lower()

    def test_french_ordinal_2e(self):
        result = normalize_numbers("le 2e chapitre", "fr")
        assert "2e" not in result


# ---------------------------------------------------------------------------
# Phoneme markup protection
# ---------------------------------------------------------------------------

class TestPhonemeMarkup:
    def test_number_inside_markup_untouched(self):
        # The number inside a phoneme markup span must NOT be converted.
        markup = "[42](/fɔːrtiˈtuː/)"
        text = f"say {markup} aloud"
        result = normalize_numbers(text, "en")
        assert markup in result  # markup preserved verbatim

    def test_number_outside_markup_converted(self):
        markup = "[word](/phonemes/)"
        text = f"chapter 3 and {markup}"
        result = normalize_numbers(text, "es")
        assert "3" not in result.replace(markup, "")
        assert markup in result


# ---------------------------------------------------------------------------
# Edge cases: numbers that should NOT be converted
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_version_string_untouched(self):
        # "v1.2.3" — preceded by a letter, must not be touched
        text = "BTS v1.2.3 is great"
        result = normalize_numbers(text, "en")
        assert "v1.2.3" in result

    def test_number_at_start_of_sentence(self):
        result = normalize_numbers("3 gatos en el tejado", "es")
        assert "3" not in result
        assert "tres" in result.lower()

    def test_number_at_end_of_sentence(self):
        result = normalize_numbers("el año fue el 2024", "es")
        assert "2024" not in result

    def test_empty_text(self):
        assert normalize_numbers("", "es") == ""

    def test_no_numbers(self):
        text = "no hay números aquí"
        assert normalize_numbers(text, "es") == text

    def test_unknown_lang_fallback(self):
        # Unknown language codes should not crash; text returned unchanged.
        text = "value 42"
        result = normalize_numbers(text, "xx")
        assert result == text


# ---------------------------------------------------------------------------
# prepare_text: integration with PronunciationDictionary
# ---------------------------------------------------------------------------

class TestPrepareText:
    def test_number_then_dictionary(self):
        # Number is expanded first, then the dictionary can match the word.
        lexicon = PronunciationDictionary({"tres": "TRES"})
        result = prepare_text("hay 3 gatos", "es", lexicon)
        # "3" → "tres" → "TRES" via dictionary
        assert "TRES" in result

    def test_dictionary_then_number_order(self):
        # Dictionary entries that ARE numbers should NOT apply before expansion.
        # (This checks that the dictionary does not replace a digit before n2w.)
        lexicon = PronunciationDictionary({"42": "cuarenta y dos"})
        result = prepare_text("hay 42 gatos", "es", lexicon)
        # num2words already converted 42 → "cuarenta y dos", dict is a no-op.
        assert "42" not in result

    def test_empty_lexicon_passthrough(self):
        result = prepare_text("chapter 3", "en", _empty_lexicon())
        assert "3" not in result
        assert "three" in result.lower()

    def test_zh_prepare_text_unchanged(self):
        text = "有3只猫"
        result = prepare_text(text, "zh", _empty_lexicon())
        assert result == text
