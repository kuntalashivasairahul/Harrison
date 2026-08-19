#!/bin/bash
set -e

# Create a local Python environment from any clone location.

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=== Checking Python 3.12 availability ==="
PYTHON_EXE=""

if command -v python3.12 &> /dev/null; then
    PYTHON_EXE="python3.12"
elif [ -f "/opt/homebrew/bin/python3.12" ]; then
    PYTHON_EXE="/opt/homebrew/bin/python3.12"
elif [ -f "/usr/local/bin/python3.12" ]; then
    PYTHON_EXE="/usr/local/bin/python3.12"
fi

if [ -z "$PYTHON_EXE" ]; then
    echo "Python 3.12 was not found. Install it, then re-run this script."
    exit 1
else
    echo "Using existing Python 3.12: $PYTHON_EXE"
fi

echo "=== Creating virtual environment .venv312 ==="
if [ -d ".venv312" ]; then
    echo "Virtual environment .venv312 already exists."
else
    "$PYTHON_EXE" -m venv .venv312
    echo "Virtual environment .venv312 created."
fi

echo "=== Upgrading pip and installing requirements ==="
.venv312/bin/python -m pip install --upgrade pip
.venv312/bin/python -m pip install -r backend/requirements.txt

echo "=== Environment Setup Complete! ==="
.venv312/bin/python --version
