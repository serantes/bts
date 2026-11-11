"""Exceptions and one-time warnings."""

from __future__ import annotations

import sys


class TTSError(Exception):
    """Error raised while synthesizing or playing audio."""


class PlayerNotFoundError(TTSError):
    """Raised when ``ffplay`` is not installed (fatal for playback)."""


class GpuMemoryError(TTSError):
    """Raised when the GPU has no free memory for the model."""


_WARNED: set = set()


def warn_once(message: str) -> None:
    """Print a warning to standard error the first time it is raised."""
    if message not in _WARNED:
        _WARNED.add(message)
        print(f"Warning: {message}", file=sys.stderr)
