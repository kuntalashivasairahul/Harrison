#!/bin/bash
set -e

# setup_env.sh
# Check if python3.12 is available, install via brew if missing, and create a clean venv.

WORKSPACE_DIR="/Users/shivasairahulkuntala/Developer/AI_Projects/nlp_models/Harrison"
cd "$WORKSPACE_DIR"

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
    echo "Python 3.12 not found. Installing python@3.12 via Homebrew..."
    if command -v brew &> /dev/null; then
        brew install python@3.12
        PYTHON_EXE="/opt/homebrew/bin/python3.12"
    else
        echo "Error: Homebrew is not installed and Python 3.12 was not found. Please install Python 3.12 manually."
        exit 1
    fi
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
.venv312/bin/pip install --upgrade pip
.venv312/bin/pip install -r backend/requirements.txt

echo "=== Environment Setup Complete! ==="
.venv312/bin/python --version
