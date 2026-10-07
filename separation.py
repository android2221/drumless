"""Audio decoding, drum removal (Meta's htdemucs model, run locally) and encoding."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np

SR = 44100
Progress = Callable[[str, float], None]  # (stage label, 0..1)


# ---------------------------------------------------------------- ffmpeg helpers

FFMPEG_INSTALL = "brew install ffmpeg" if sys.platform == "darwin" else "sudo apt-get install -y ffmpeg"


def ffmpeg_bin() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError(f"ffmpeg not found. Install it with: {FFMPEG_INSTALL}")
    return path


def probe_tags(path: Path) -> dict:
    """Return lower-cased format tags (title, artist, album...) and whether there's cover art."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {}
    out = subprocess.run(
        [ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    try:
        info = json.loads(out.stdout or "{}")
    except json.JSONDecodeError:
        return {}
    tags = {k.lower(): v for k, v in (info.get("format", {}).get("tags") or {}).items()}
    # Some formats (ogg/flac) keep tags on the stream instead.
    for s in info.get("streams", []):
        if s.get("codec_type") == "audio":
            for k, v in (s.get("tags") or {}).items():
                tags.setdefault(k.lower(), v)
    tags["_has_art"] = any(
        s.get("codec_type") == "video" and (s.get("disposition") or {}).get("attached_pic")
        for s in info.get("streams", [])
    )
    return tags


def decode(path: Path) -> np.ndarray:
    """Decode any format ffmpeg understands to float32 stereo, shape (2, n)."""
    proc = subprocess.run(
        [ffmpeg_bin(), "-v", "error", "-i", str(path), "-vn", "-f", "f32le",
         "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(SR), "pipe:1"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Could not decode audio: {proc.stderr.decode(errors='ignore').strip()[-300:]}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size < SR:  # under half a second of stereo
        raise RuntimeError("File has no usable audio.")
    return audio.reshape(-1, 2).T.copy()


PERSONAL_TAGS = ("account_id", "purchase_date", "owner", "ownr", "apID", "purd", "itunes_cddb_1")

CODECS = {
    "m4a": ["-c:a", "aac", "-b:a", "256k"],
    "flac": ["-c:a", "flac"],
    "wav": ["-c:a", "pcm_s16le"],
    "mp3": ["-c:a", "libmp3lame", "-b:a", "320k"],
}

# Output format for each source extension when the format setting is "auto".
AUTO_FORMATS = {".mp3": "mp3", ".flac": "flac", ".wav": "wav", ".aif": "wav", ".aiff": "wav"}


def resolve_format(fmt: str, source_name: str) -> str:
    """Turn "auto" into the source file's own format (M4A for anything we can't write back)."""
    if fmt != "auto":
        return fmt
    return AUTO_FORMATS.get(Path(source_name).suffix.lower(), "m4a")


def encode(audio: np.ndarray, source: Path, dest: Path, fmt: str, title: str | None) -> None:
    """Write audio to dest, copying tags (and cover art when the format allows) from source."""
    pcm = np.clip(audio, -1.0, 1.0).T.astype(np.float32).tobytes()
    base = [ffmpeg_bin(), "-v", "error", "-y",
            "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "pipe:0",
            "-i", str(source), "-map", "0:a", "-map_metadata", "1"]
    meta = ["-metadata", f"title={title}"] if title else []
    # Don't carry the purchaser's Apple ID, name or purchase date into the copies.
    for key in PERSONAL_TAGS:
        meta += ["-metadata", f"{key}="]
    art = ["-map", "1:v?", "-c:v", "copy", "-disposition:v", "attached_pic"] if fmt in ("m4a", "mp3", "flac") else []
    tail = CODECS[fmt] + meta + (["-movflags", "+faststart"] if fmt == "m4a" else []) + [str(dest)]

    proc = subprocess.run(base + art + tail, input=pcm, capture_output=True)
    if proc.returncode != 0 and art:
        # Some cover-art formats can't be carried across containers; retry without it.
        proc = subprocess.run(base + tail, input=pcm, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Could not encode output: {proc.stderr.decode(errors='ignore').strip()[-300:]}")


# ---------------------------------------------------------------- drum removal

def _torch_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"  # Apple Silicon GPU
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


_model = None


def remove_drums(audio: np.ndarray, progress: Progress | None = None) -> np.ndarray:
    """Run htdemucs in-process and return everything except the drum stem.

    Calls the Demucs library directly (not its CLI) so we never touch torchaudio's
    file I/O, which newer torchaudio releases removed. The model (~80 MB) downloads
    once on first use and is cached by torch, then stays loaded between songs.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    import demucs.apply as dapply
    from demucs.pretrained import get_model

    global _model
    if _model is None:
        if progress:
            progress("Loading AI model", 0.08)
        _model = get_model("htdemucs")
        _model.eval()

    # Report Demucs' internal chunk loop to the UI instead of a terminal progress bar.
    class _Progress:
        @staticmethod
        def tqdm(items, **_):
            items = list(items)
            for i, item in enumerate(items):
                if progress:
                    progress("Separating (AI)", 0.1 + 0.75 * i / max(len(items), 1))
                yield item

    wav = torch.from_numpy(audio)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    real_tqdm, dapply.tqdm = dapply.tqdm, _Progress
    try:
        with torch.no_grad():
            sources = dapply.apply_model(_model, ((wav - mean) / std)[None], device=_torch_device(),
                                         shifts=1, split=True, overlap=0.25, progress=True)[0]
    finally:
        dapply.tqdm = real_tqdm
    sources = sources * std + mean
    keep = [i for i, name in enumerate(_model.sources) if name != "drums"]
    return sources[keep].sum(0).cpu().numpy().astype(np.float32)
