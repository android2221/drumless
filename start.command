#!/bin/bash
# Double-click in Finder (or run ./start.command) to launch Drumless. Works on macOS and Ubuntu.
# First run sets everything up; after that it starts in a couple of seconds.
set -e
cd "$(dirname "$0")"

HOST="${DRUMLESS_HOST:-127.0.0.1}"
PORT="${DRUMLESS_PORT:-8765}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    echo "Installing ffmpeg…"; brew install ffmpeg
  elif command -v apt-get >/dev/null 2>&1; then
    echo "Drumless needs ffmpeg. Run: sudo apt-get install -y ffmpeg"
    exit 1
  else
    echo "Drumless needs ffmpeg. Install Homebrew (https://brew.sh), then run: brew install ffmpeg"
    exit 1
  fi
fi

# .venv/.deps-ok marks a finished install, so a failed first run is retried instead of
# leaving a half-built environment behind.
if [ ! -f .venv/.deps-ok ] || [ requirements.txt -nt .venv/.deps-ok ]; then
  echo "Setting up Python environment…"
  if [ ! -x .venv/bin/pip ]; then
    rm -rf .venv
    if ! python3 -m venv .venv; then
      rm -rf .venv
      echo "Couldn't create a Python environment. On Ubuntu run: sudo apt-get install -y python3-venv"
      exit 1
    fi
  fi
  .venv/bin/pip install --upgrade pip -q
  # --prefer-binary: take a slightly older prebuilt wheel over compiling the newest release
  # (numba/llvmlite no longer ship wheels for Intel-mac Python, and building them fails).
  .venv/bin/pip install --prefer-binary -r requirements.txt -q
  touch .venv/.deps-ok
fi

# Optional AI mode: ./start.command --with-ai  (one-time ~2 GB download for PyTorch + model)
if [ "$1" = "--with-ai" ]; then
  echo "Installing Demucs (AI mode)…"
  .venv/bin/pip install --prefer-binary demucs soundfile -q
fi

# Open the browser when there's a desktop to open it on (skipped on a headless server).
if [ "$(uname)" = "Darwin" ]; then
  (sleep 2 && open "http://127.0.0.1:$PORT") &
elif [ -n "$DISPLAY$WAYLAND_DISPLAY" ] && command -v xdg-open >/dev/null 2>&1; then
  (sleep 2 && xdg-open "http://127.0.0.1:$PORT" >/dev/null 2>&1) &
fi

export DRUMLESS_HOST="$HOST" DRUMLESS_PORT="$PORT"
exec .venv/bin/python app.py
