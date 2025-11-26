#!/usr/bin/env bash
# run with: source path-to/setup-server.sh

set -e

# Name or path of your venv
VENV_DIR="./server"

# Detect if a venv is already active
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "Warning: A virtual environment is already active: $VIRTUAL_ENV"
    echo "Skipping creation/activation to avoid nested venvs."
    return 0 2>/dev/null || exit 0
fi

# Create venv if it doesn't exist
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# Activate the venv
echo "Activating virtual environment..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Virtual environment is now active."
echo "Python: $(python --version)"
echo "pip: $(pip --version)"
pip install ansible-runner
