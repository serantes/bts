"""Spelling analysis (--analyze)."""

import argparse

import pytest

from bts import analysis, errors


class FakeAnalyzer(analysis.WordAnalyzer):
    known = {"hola", "mundo", "the"}

    def __init__(self):
        self.closed = False

    def words(self, text):
        return [w for w in analysis.WORD_RE.findall(text) if len(w) > 1]

    def is_known(self, word):
        return word.lower() in self.known

    def close(self):
        self.closed = True


def test_base_analyzer_is_abstract():
    base = analysis.WordAnalyzer()
    with pytest.raises(NotImplementedError):
        base.words("x")
    with pytest.raises(NotImplementedError):
        base.is_known("x")
    base.close()


def test_analyze_text_lists_unknown_once_sorted():
    text = "hola Zorg zorg ZORG mundo Abra zorg"
    assert analysis.analyze_text(text, FakeAnalyzer()) == ["Abra", "zorg"]


def test_analyze_text_tie_keeps_first_spelling():
    assert analysis.analyze_text("Zorg zorg", FakeAnalyzer()) == ["Zorg"]


def test_word_regex_handles_apostrophes_hyphens_and_ignores_digits():
    words = analysis.WORD_RE.findall("don't bien-être l’homme 123 a_b")
    assert words == ["don't", "bien-être", "l’homme", "a", "b"]


def test_default_dict_path():
    assert analysis.default_dict_path("/a/b/book.epub") == "book.dict"


def test_register_and_get_analyzer(monkeypatch):
    monkeypatch.setitem(analysis.ANALYZER_FACTORIES, "ja", lambda code: "A")
    assert analysis.get_analyzer("JA") == "A"
    analysis.register_analyzer("zz", lambda code: code)
    try:
        assert analysis.get_analyzer("zz") == "zz"
    finally:
        analysis.ANALYZER_FACTORIES.pop("zz")


def test_get_analyzer_unsupported():
    with pytest.raises(errors.TTSError, match="not available"):
        analysis.get_analyzer("zh")


# --- Hunspell ----------------------------------------------------------------


def test_find_hunspell_dictionary_uses_dicpath_and_preference(
    monkeypatch, tmp_path
):
    for name in ("es_MX", "es_ES"):
        (tmp_path / f"{name}.aff").write_text("")
        (tmp_path / f"{name}.dic").write_text("0")
    monkeypatch.setenv("DICPATH", str(tmp_path))
    affix, dic = analysis.find_hunspell_dictionary("es")
    assert affix.endswith("es_ES.aff") and dic.endswith("es_ES.dic")


def test_find_hunspell_dictionary_needs_both_files(monkeypatch, tmp_path):
    (tmp_path / "es_ES.aff").write_text("")
    monkeypatch.setenv("DICPATH", str(tmp_path))
    monkeypatch.setattr(analysis, "HUNSPELL_DIRS", ())
    with pytest.raises(errors.TTSError, match="No Hunspell dictionary"):
        analysis.find_hunspell_dictionary("es")


def make_hunspell(known, encoding="utf-8"):
    analyzer = object.__new__(analysis.HunspellAnalyzer)
    analyzer._encoding = encoding
    analyzer._spell = lambda word: word in known
    return analyzer


def test_hunspell_is_known_with_parts():
    analyzer = make_hunspell({"bien", "être", "homme", "don"})
    assert analyzer.is_known("bien-être")
    assert analyzer.is_known("l’homme")  # single-letter part accepted
    assert not analyzer.is_known("bien-xyz")
    assert not analyzer.is_known("xyz")


def test_hunspell_words_ignore_single_letters():
    analyzer = make_hunspell(set())
    assert analyzer.words("a bc d ef") == ["bc", "ef"]


def test_hunspell_spell_unencodable_word():
    analyzer = object.__new__(analysis.HunspellAnalyzer)
    analyzer._encoding = "ascii"
    analyzer._lib = None
    assert analysis.HunspellAnalyzer._spell(analyzer, "ñandú") is False


def test_hunspell_missing_library(monkeypatch):
    monkeypatch.setattr(analysis.ctypes.util, "find_library",
                        lambda name: None)
    monkeypatch.setattr(analysis, "HUNSPELL_SONAMES", ("libnope.so.0",))
    with pytest.raises(errors.TTSError, match="libhunspell"):
        analysis.HunspellAnalyzer._load_library()


# --- phonemes and word list --------------------------------------------------


def test_add_phonemes(monkeypatch, fake_pipeline):
    monkeypatch.setattr(analysis, "get_g2p_pipeline",
                        lambda code: fake_pipeline)
    entries, missing = analysis.add_phonemes(["Zorg", " "], "es")
    assert entries[0] == ("Zorg", "[Zorg](/ZORG/)")
    assert entries[1] == (" ", " ") and missing == 1


def test_add_phonemes_uses_requested_language(monkeypatch, fake_pipeline):
    seen = []
    monkeypatch.setattr(
        analysis, "get_g2p_pipeline",
        lambda code: seen.append(code) or fake_pipeline
    )
    analysis.add_phonemes(["x"], "es")
    analysis.add_phonemes(["x"], "us")
    assert seen == ["e", "a"]


def test_add_phonemes_forbidden_chars_and_errors(monkeypatch, fake_pipeline):
    fake_pipeline.g2p = lambda text: ("a/b", None)
    monkeypatch.setattr(analysis, "get_g2p_pipeline",
                        lambda code: fake_pipeline)
    entries, missing = analysis.add_phonemes(["w"], "es")
    assert entries == [("w", "w")] and missing == 1

    fake_pipeline.g2p = lambda text: (_ for _ in ()).throw(ValueError())
    assert analysis.add_phonemes(["w"], "es")[1] == 1


def test_write_word_list_and_never_overwrites(tmp_path):
    path = str(tmp_path / "w.dict")
    analysis.write_word_list(path, [("a", "a"), ("b", "[b](/x/)")])
    assert open(path, encoding="utf-8").read() == "a|a\nb|[b](/x/)\n"
    with pytest.raises(errors.TTSError, match="already exists"):
        analysis.write_word_list(path, [])


def test_write_word_list_unwritable(tmp_path):
    with pytest.raises(errors.TTSError, match="Cannot write"):
        analysis.write_word_list(str(tmp_path / "no" / "w.dict"), [])


# --- run_analysis ------------------------------------------------------------


def analysis_args(**kw):
    # Mirrors argparse defaults: ipa=None when flag absent, "" when bare,
    # "<code>" when --ipa LANG is supplied.
    values = dict(
        epub=None, text_file=None, dictionary=None, lang="es", ipa=None
    )
    values.update(kw)
    return argparse.Namespace(**values)


def test_run_analysis_text(monkeypatch, tmp_path, capsys):
    analyzer = FakeAnalyzer()
    monkeypatch.setattr(analysis, "get_analyzer", lambda lang: analyzer)
    source = tmp_path / "s.txt"
    source.write_text("hola Zorg mundo", encoding="utf-8")
    target = tmp_path / "out.dict"
    args = analysis_args(text_file=str(source), dictionary=str(target))
    assert analysis.run_analysis(args) == 0
    assert target.read_text(encoding="utf-8") == "Zorg|Zorg\n"
    assert analyzer.closed


def test_run_analysis_refuses_existing_target_before_work(monkeypatch,
                                                          tmp_path):
    def boom(lang):
        raise AssertionError("must fail before creating the analyzer")

    monkeypatch.setattr(analysis, "get_analyzer", boom)
    target = tmp_path / "out.dict"
    target.write_text("keep")
    args = analysis_args(text_file="x.txt", dictionary=str(target))
    with pytest.raises(errors.TTSError, match="already exists"):
        analysis.run_analysis(args)
    assert target.read_text() == "keep"


def test_run_analysis_closes_analyzer_on_error(monkeypatch, tmp_path):
    analyzer = FakeAnalyzer()
    monkeypatch.setattr(analysis, "get_analyzer", lambda lang: analyzer)
    args = analysis_args(
        text_file=str(tmp_path / "missing.txt"),
        dictionary=str(tmp_path / "o.dict"),
    )
    with pytest.raises(errors.TTSError):
        analysis.run_analysis(args)
    assert analyzer.closed


def test_run_analysis_with_ipa_no_lang(monkeypatch, tmp_path, fake_pipeline):
    """--ipa without LANG uses --lang for the phoneme lookup."""
    seen_codes = []

    def capture_pipeline(code):
        seen_codes.append(code)
        return fake_pipeline

    monkeypatch.setattr(analysis, "get_analyzer", lambda lang: FakeAnalyzer())
    monkeypatch.setattr(analysis, "get_g2p_pipeline", capture_pipeline)
    source = tmp_path / "s.txt"
    source.write_text("Zorg", encoding="utf-8")
    target = tmp_path / "o.dict"
    # ipa="" means bare --ipa with no LANG argument.
    args = analysis_args(
        text_file=str(source), dictionary=str(target), lang="es", ipa=""
    )
    analysis.run_analysis(args)
    assert target.read_text(encoding="utf-8") == "Zorg|[Zorg](/ZORG/)\n"
    # The pipeline must have been created with the internal code for 'es'.
    from bts.languages import SUPPORTED_LANGUAGES
    assert seen_codes == [SUPPORTED_LANGUAGES["es"]["internal"]]


def test_run_analysis_with_ipa_lang_override(monkeypatch, tmp_path,
                                             fake_pipeline):
    """--ipa LANG uses the override language for phoneme lookup, not --lang."""
    seen_codes = []

    def capture_pipeline(code):
        seen_codes.append(code)
        return fake_pipeline

    monkeypatch.setattr(analysis, "get_analyzer", lambda lang: FakeAnalyzer())
    monkeypatch.setattr(analysis, "get_g2p_pipeline", capture_pipeline)
    source = tmp_path / "s.txt"
    source.write_text("Smith", encoding="utf-8")
    target = tmp_path / "o.dict"
    # Analyze Spanish text but generate English IPA for unknown proper nouns.
    args = analysis_args(
        text_file=str(source), dictionary=str(target), lang="es", ipa="us"
    )
    analysis.run_analysis(args)
    # The IPA pipeline must have been requested for 'us', not 'es'.
    from bts.languages import SUPPORTED_LANGUAGES
    assert seen_codes == [SUPPORTED_LANGUAGES["us"]["internal"]]
    # The entry is still generated correctly.
    assert target.read_text(encoding="utf-8") == "Smith|[Smith](/SMITH/)\n"


def test_run_analysis_ipa_message_shows_lang_only_when_different(
    monkeypatch, tmp_path, fake_pipeline, capsys
):
    """The '[ANALYSIS] Looking up...' line shows the IPA language only when
    it differs from --lang."""
    monkeypatch.setattr(analysis, "get_analyzer", lambda lang: FakeAnalyzer())
    monkeypatch.setattr(
        analysis, "get_g2p_pipeline", lambda code: fake_pipeline
    )
    source = tmp_path / "s.txt"
    source.write_text("Zorg", encoding="utf-8")

    # Case 1: bare --ipa (ipa="") — no language hint in the message.
    target1 = tmp_path / "o1.dict"
    args1 = analysis_args(
        text_file=str(source), dictionary=str(target1), lang="es", ipa=""
    )
    analysis.run_analysis(args1)
    out1 = capsys.readouterr().out
    assert "IPA language" not in out1

    # Case 2: --ipa LANG with a different code — hint appears.
    target2 = tmp_path / "o2.dict"
    args2 = analysis_args(
        text_file=str(source), dictionary=str(target2), lang="es", ipa="us"
    )
    analysis.run_analysis(args2)
    out2 = capsys.readouterr().out
    assert "IPA language: 'us'" in out2


def test_run_analysis_no_ipa_flag_skips_phonemes(
    monkeypatch, tmp_path
):
    """Without --ipa the word list contains plain word|word entries."""
    monkeypatch.setattr(analysis, "get_analyzer", lambda lang: FakeAnalyzer())
    source = tmp_path / "s.txt"
    source.write_text("Zorg", encoding="utf-8")
    target = tmp_path / "o.dict"
    # ipa=None is what argparse sets when --ipa is not given at all.
    args = analysis_args(
        text_file=str(source), dictionary=str(target), ipa=None
    )
    analysis.run_analysis(args)
    assert target.read_text(encoding="utf-8") == "Zorg|Zorg\n"
