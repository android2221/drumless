#!/bin/bash
# Double-click in Finder (or run ./start.command) to launch Drumless.
# First run sets everything up; after that it starts in a couple of seconds.
set -e
cd "$(dirname "$0")"

if ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    echo "Installing ffmpeg…"; brew install ffmpeg
  else
    echo "Drumless needs ffmpeg. Install Homebrew (https://brew.sh), then run: brew install ffmpeg"
    exit 1
  fi
fi

if [ ! -d .venv ]; then
  echo "First run: setting up Python environment…"
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip -q
  .venv/bin/pip install -r requirements.txt -q
fi

# Optional AI mode: ./start.command --with-ai  (one-time ~2 GB download for PyTorch + model)
if [ "$1" = "--with-ai" ]; then
  echo "Installing Demucs (AI mode)…"
  .venv/bin/pip install demucs soundfile -q
fi

(sleep 2 && open "http://127.0.0.1:8765") &
exec .venv/bin/python app.py
