#!/usr/bin/env bash

# Python 3.10-3.12 required (3.12 preferred).

set -e
cd "$(dirname "$0")/.."

echo "Creating virtual environment for BTS"

[ -d .venv ] || python3.12 -m venv .venv "$@"

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[all]"

# Load unidic dictionary (Japanese support).
echo "Downloading unidic dictionary"
.venv/bin/python -m unidic download

echo "Initialization finished"
