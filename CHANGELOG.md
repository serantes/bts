# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.1.0] – Unreleased

Initial public release. The project was reorganised from a single-file script
into a properly packaged Python application.

### Added

- `pyproject.toml` with `src/` layout, entry point `bts = "bts.cli:main"`,
  optional extras (`ja`, `zh`, `all`, `dev`), and `ruff`/`pytest` config.
- `--version` flag.
- `--ipa [LANG]`: `--analyze` now accepts an optional language code for the
  IPA phoneme lookup, independent of `--lang` (e.g. `--lang es --ipa us` for
  English proper nouns inside a Spanish book).
- `-` as text argument to read standard input explicitly.
- Non-blocking stdin detection (`_stdin_has_data`) so that BTS never hangs
  when run from cron or in the background without a pipe.
- `--display` flag for real-time word highlighting during playback.
- Comprehensive unit-test suite (180 tests, no GPU or network required).
  All heavy dependencies (torch, kokoro, ffplay, ffmpeg, hunspell, ebooklib)
  are replaced by lightweight mocks in `tests/conftest.py`.
- `docs/usage.md` — full Spanish user manual.
- `LICENSE` (MIT).

### Changed

- Monolithic `bts.py` split into focused modules under `src/bts/`:
  `errors`, `constants`, `languages`, `phonemes`, `dictionary`, `device`,
  `audio`, `textio`, `synthesis`, `epub`, `analysis`, `cli`.
- Section title and body now share a single ffplay/ffmpeg stream, eliminating
  the gap between the title announcement and the body text.
- Section title speed scales with `--speed` (factor `TITLE_SPEED = 0.8`).
- Lazy loading of `torch` and `kokoro`: startup commands (`--help`,
  `--list-voices`, `--section-list`, …) no longer load model weights.

### Fixed

- `--analyze --ipa` always generated phonemes for English, ignoring `--lang`.
- `--speed` accepted `nan` and `inf`; they are now rejected with a clear error.
- `positive_float` now raises `ArgumentTypeError` with `from None` to avoid
  leaking the internal `ValueError` traceback.
