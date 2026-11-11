"""BTS - Book To Speech.

Local neural text-to-speech (Kokoro-82M) for text, text files and EPUB books.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bts")
except PackageNotFoundError:  # running from a source tree, not installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
