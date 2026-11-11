"""Allow ``python -m bts``."""

import sys

from bts.cli import main

if __name__ == "__main__":
    sys.exit(main())
