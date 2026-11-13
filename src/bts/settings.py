"""Per-request settings shared by synthesis and the daemon."""

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field

from bts.constants import (
    AUDIO_FORMATS,
    GPU_MIN_FREE_MIB,
    IGNORED_STRINGS,
    MAX_CHUNK_CHARS,
    MIN_SECTION_CHARS,
    PAUSE_MARKERS,
    PHONEME_CHUNK_CHARS,
    REPO_ID,
    TITLE_PAUSE_SECONDS,
    TITLE_SPEED,
)
from bts.errors import TTSError


@dataclass(frozen=True)
class Settings:
    default_format: str = "mp3"
    gpu_min_free_mib: int = GPU_MIN_FREE_MIB
    ignored_strings: tuple[str, ...] = tuple(IGNORED_STRINGS)
    max_chunk_chars: int = MAX_CHUNK_CHARS
    min_section_chars: int = MIN_SECTION_CHARS
    pause_markers: dict[str, float] = field(
        default_factory=lambda: dict(PAUSE_MARKERS)
    )
    phoneme_chunk_chars: int = PHONEME_CHUNK_CHARS
    repo_id: str = REPO_ID
    title_pause_seconds: float = TITLE_PAUSE_SECONDS
    title_speed: float = TITLE_SPEED


_DEFAULTS = Settings()
_current_settings: ContextVar[Settings] = ContextVar(
    "bts_settings", default=_DEFAULTS
)


def current_settings() -> Settings:
    return _current_settings.get()


def settings_values() -> dict:
    return asdict(_DEFAULTS)


def _finite_number(value) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def validate_setting(key: str, value):
    """Validate a YAML or daemon setting and return its normalized value."""
    valid = False
    if key == "default_format":
        valid = isinstance(value, str) and value in AUDIO_FORMATS
    elif key == "repo_id":
        valid = isinstance(value, str) and bool(value.strip())
    elif key == "ignored_strings":
        valid = isinstance(value, (list, tuple)) and all(
            isinstance(item, str) and bool(item) for item in value
        )
        if valid:
            return tuple(value)
    elif key == "pause_markers":
        valid = isinstance(value, dict) and all(
            isinstance(marker, str) and bool(marker)
            and _finite_number(duration) and duration >= 0
            for marker, duration in value.items()
        )
        if valid:
            return dict(value)
    elif key in {
        "gpu_min_free_mib", "max_chunk_chars", "min_section_chars",
        "phoneme_chunk_chars",
    }:
        minimum = 0 if key in {"gpu_min_free_mib", "min_section_chars"} else 1
        valid = (
            isinstance(value, int) and not isinstance(value, bool)
            and value >= minimum
        )
    elif key in {"title_pause_seconds", "title_speed"}:
        valid = (
            _finite_number(value)
            and (value >= 0 if key == "title_pause_seconds" else value > 0)
        )
    if not valid:
        raise TTSError(f"Invalid value for '{key}': {value!r}.")
    return value


@contextmanager
def use_settings(values: dict):
    """Apply validated settings in the current execution context."""
    if not isinstance(values, dict):
        raise TTSError("Settings must be a mapping.")
    defaults = settings_values()
    for key, value in values.items():
        if key not in defaults:
            raise TTSError(f"Unknown setting {key!r}.")
        defaults[key] = validate_setting(key, value)
    token = _current_settings.set(Settings(**defaults))
    try:
        yield
    finally:
        _current_settings.reset(token)
