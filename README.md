# Drumless

Drop songs into a web page; get copies back without drums. Runs entirely on your own machine
(macOS or Ubuntu).

## Start it

1. Double-click **start.command** (or run `./start.command` in Terminal from this folder).
   The first run installs ffmpeg (via Homebrew) and the Python packages, including PyTorch
   (about 1 GB). That takes a few minutes.
2. Your browser opens **http://127.0.0.1:8765**.
3. Click **Choose…** to pick a save folder, then drag songs or whole folders onto the page.

If macOS blocks the double-click ("unidentified developer"), right-click → Open once, or run
`chmod +x start.command` first.

## Ubuntu

```
sudo apt-get install -y ffmpeg python3-venv
./start.command
```

The page is served on `127.0.0.1:8765`, so on a remote server reach it through an SSH tunnel:
`ssh -L 8765:127.0.0.1:8765 you@server`, then open http://127.0.0.1:8765 on your own computer.
Files are saved on the server; type the save folder's path into the page (the folder chooser
and "Show in Finder" are macOS-only).

`DRUMLESS_HOST` and `DRUMLESS_PORT` change the address, e.g. `DRUMLESS_HOST=0.0.0.0 ./start.command`.
There is no login, and anyone who can reach the page can write files on the server, so only
do that on a network you trust.

## How it works

Drums are separated by Meta's **htdemucs** model (Demucs), run locally; everything except the
drum stem is mixed back together. It uses the Mac's GPU (or an NVIDIA GPU on Linux) when there
is one, and falls back to the CPU, which is much slower. On a Mac with a GPU expect roughly
5 seconds per minute of music.

The model (~80 MB) downloads once, on the first song, from `dl.fbaipublicfiles.com` and is cached
in `~/.cache/torch/hub/checkpoints/`. After that no network access is needed.

## Where files go

`<save folder>/<Artist>/<Album>/<original name> (No Drums).m4a`

Title, artist, album and cover art are copied over; the title gets " (No Drums)" appended so the
copies are easy to tell apart if you add them to the Music app.

## Never processing a song twice

The save folder holds a small `.drumless-manifest.json` that records a fingerprint (SHA-256) of every
source file processed into it. A song is skipped when:

- its fingerprint is already in the manifest (even if the file was renamed or moved),
- it's already waiting in the queue,
- it's itself a Drumless output, or
- a file with the same output name already exists.

To redo a song, delete its output file and drop it again.

## Settings

Stored in `~/.drumless/config.json`. Formats: Auto (same as each source file; M4A when the source is
something else, like OGG), M4A (AAC 256k), FLAC, WAV, MP3 (320k).

## Files

- `app.py` — local web server, job queue, duplicate checks
- `separation.py` — decoding, drum removal (Demucs), tag-preserving export
- `static/index.html` — the drag-and-drop page
