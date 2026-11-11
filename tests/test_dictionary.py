"""Pronunciation dictionary and phoneme markup."""

import pytest

from bts import dictionary, errors, phonemes


def test_parse_phonemes():
    assert phonemes.parse_phonemes("mister") is None
    assert phonemes.parse_phonemes("[mˈɪstəɹ]") == "mˈɪstəɹ"
    assert phonemes.parse_phonemes("  [ab]  ") == "ab"
    assert phonemes.parse_phonemes("[]") == ""
    assert phonemes.parse_phonemes("[") is None
    with pytest.raises(ValueError):
        phonemes.parse_phonemes("[a/b]")
    with pytest.raises(ValueError):
        phonemes.parse_phonemes("[a(b)]")


def test_phoneme_markup_label():
    assert phonemes.phoneme_markup("Mr.", "x") == "[Mr](/x/)"
    assert phonemes.phoneme_markup("...", "x") == "[x](/x/)"


def test_strip_phoneme_markup():
    assert phonemes.strip_phoneme_markup("a [mr](/mˈɪ/) b") == "a mr b"


def test_whole_words_only():
    lexicon = dictionary.PronunciationDictionary({"mr": "mister"})
    assert lexicon.apply("mr Smith, mrs Smith") == "mister Smith, mrs Smith"


def test_case_sensitivity():
    assert dictionary.PronunciationDictionary({"mr": "m"}).apply("Mr") == "Mr"
    lexicon = dictionary.PronunciationDictionary({"mr": "m"}, ignore_case=True)
    assert lexicon.apply("Mr MR mr") == "m m m"


def test_longest_entry_wins():
    lexicon = dictionary.PronunciationDictionary({"a": "1", "a b": "2"})
    assert lexicon.apply("a b a") == "2 1"


def test_punctuation_edges_need_no_boundary():
    lexicon = dictionary.PronunciationDictionary({"Mr.": "Mister"})
    assert lexicon.apply("Mr.Smith and Mr. Jones") == "MisterSmith and Mister Jones"


def test_phoneme_replacement_and_removal():
    lexicon = dictionary.PronunciationDictionary({"mr": "[mˈɪ]", "x": "[]"})
    assert lexicon.apply("mr x") == "[mr](/mˈɪ/) "


def test_special_regex_characters_are_escaped():
    lexicon = dictionary.PronunciationDictionary({"c++": "cplusplus"})
    assert lexicon.apply("I like c++ a lot") == "I like cplusplus a lot"


def test_empty_dictionary_is_noop():
    lexicon = dictionary.PronunciationDictionary({})
    assert len(lexicon) == 0
    assert lexicon.apply("hello") == "hello"


def test_load_dictionary(tmp_path, capsys):
    path = tmp_path / "d.txt"
    path.write_text(
        "\ufeff# comment\n\nmr | mister \nbad line\n|nothing\n"
        "x|[a/b]\nph|[fə]\nmr|sir|ignored\n",
        encoding="utf-8",
    )
    lexicon = dictionary.load_dictionary(str(path))
    assert len(lexicon) == 2  # last 'mr' wins, 'ph' kept
    assert lexicon.apply("mr ph") == "sir [ph](/fə/)"
    err = capsys.readouterr().err
    assert err.count("line skipped") == 3


def test_load_dictionary_missing_file():
    with pytest.raises(errors.TTSError, match="Cannot read"):
        dictionary.load_dictionary("/nonexistent/file.txt")


def test_load_dictionary_not_utf8(tmp_path):
    path = tmp_path / "d.txt"
    path.write_bytes(b"\xff\xfe\x00bad|x\n\x80")
    with pytest.raises(errors.TTSError):
        dictionary.load_dictionary(str(path))


def test_load_dictionary_ignore_case(tmp_path):
    path = tmp_path / "d.txt"
    path.write_text("mr|mister\n", encoding="utf-8")
    assert dictionary.load_dictionary(str(path), True).apply("MR") == "mister"
