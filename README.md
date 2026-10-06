# Drumless

Drop songs into a web page; get copies back without drums. Runs entirely on your own machine
(macOS or Ubuntu).

## Start it

1. Double-click **start.command** (or run `./start.command` in Terminal from this folder).
   The first run installs ffmpeg (via Homebrew) and the Python packages. That takes a few minutes.
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

- **Classic (no AI)** — splits the spectrum into bass, mid and treble and removes what looks like
  a drum hit in each: sudden thumps in the bass, short broadband bursts above it. Fast (a few
  seconds per song) and needs nothing extra, but drums are only turned down, never gone, and
  plucked/strummed attacks soften. Strength trades drum removal against damage to the music:
  - *Gentle*: cleanest music, drums a little quieter.
  - *Normal*: the best balance by measurement.
  - *Aggressive*: most removal; the music gets noticeably duller.
- **AI (Demucs)** — Meta's htdemucs model, running on the Mac's GPU. Much cleaner, roughly
  20–40 s per song. Turn it on once with `./start.command --with-ai` (downloads ~2 GB of PyTorch,
  plus an ~80 MB model on the first song).

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

To redo a song (for example with AI mode), delete its output file and drop it again.

## Settings

Stored in `~/.drumless/config.json`. Formats: Auto (same as each source file; M4A when the source is
something else, like OGG), M4A (AAC 256k), FLAC, WAV, MP3 (320k).

## Files

- `app.py` — local web server, job queue, duplicate checks
- `separation.py` — decoding, the two drum-removal methods, tag-preserving export
- `static/index.html` — the drag-and-drop page
