"""Audio decoding, drum removal and encoding.

Two methods:
  - "dsp":    harmonic-percussive source separation (no AI). Fast, no extra installs.
  - "demucs": Meta's htdemucs model, run locally. Much cleaner, needs `pip install demucs`.
"""

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

# How hard the DSP method leans on removing drums: a multiplier on every band's margin.
# Higher = only clearly drum-like energy is removed (gentler, cleaner music).
STRENGTH_MARGINS = {"gentle": 2.0, "normal": 1.0, "aggressive": 0.5}

# The spectrum is split into three bands, each with the rule that scored best for it against
# songs with known drum stems (tuned by measurement, not by ear):
#   bass   - kick vs bass guitar: anything rising above the level sustained over ~0.35 s is kick.
#   mid    - snare/toms vs vocals and guitars: only broadband bursts go; cautious margin.
#   treble - hats/cymbals: broadband bursts again, wider frequency kernel, bolder margin.
BAND_EDGES_HZ = (200, 4000)
BASS = {"sustain_s": 0.35, "margin": 1.0}
MID = {"sustain_s": 0.2, "spread_hz": 300, "margin": 2.0}
TREBLE = {"sustain_s": 0.2, "spread_hz": 1200, "margin": 1.0}
CROSSFADE = 0.15  # width of the blend between bands, in log-frequency units

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


# ---------------------------------------------------------------- DSP (no AI)

def _odd(x: float) -> int:
    return max(3, int(round(x)) | 1)


def remove_drums_dsp(audio: np.ndarray, strength: str = "normal", progress: Progress | None = None) -> np.ndarray:
    """Median-filter drum removal, with a separate rule per frequency band.

    Drums show up as vertical lines in a spectrogram (broadband, short); pitched
    instruments show up as horizontal lines (narrowband, sustained). A median filter
    along time estimates what is sustained (music); one along frequency estimates
    what is broadband (drums). A soft mask removes the drum share of each bin.

    The mask is computed once from the mid (L+R) signal and applied to both
    channels so the stereo image doesn't wobble.
    """
    import librosa  # imported lazily so the server starts fast
    from scipy.ndimage import median_filter

    n_fft, hop = 2048, 512
    scale = STRENGTH_MARGINS.get(strength, STRENGTH_MARGINS["normal"])
    n = audio.shape[1]
    eps = 1e-10

    if progress:
        progress("Analysing", 0.15)
    specs = [librosa.stft(ch, n_fft=n_fft, hop_length=hop) for ch in audio]
    mag = np.abs(specs[0] + specs[1]) * 0.5
    freqs = np.arange(mag.shape[0]) * SR / n_fft

    def frames(seconds: float) -> int:
        return _odd(seconds * SR / hop)

    def band_rows(lo_hz: float, hi_hz: float, pad: int) -> tuple[slice, slice]:
        """Rows needed to filter [lo_hz, hi_hz] (padded so the edges match a full-height filter),
        and where the wanted rows sit inside that padded block."""
        rows = np.flatnonzero((freqs >= lo_hz) & (freqs <= hi_hz))
        first, last = rows[0], rows[-1] + 1
        start, stop = max(first - pad, 0), min(last + pad, mag.shape[0])
        return slice(start, stop), slice(first - start, last - start)

    def sustained(block: np.ndarray, seconds: float) -> np.ndarray:
        return np.minimum(median_filter(block, size=(1, frames(seconds)), mode="reflect"), block)

    fade = float(np.exp(CROSSFADE))
    drum = np.zeros_like(mag)  # share of each bin that is drum, 0..1

    # Bass: kick drums are narrow low thumps, so "broadband" doesn't describe them. Treat any
    # energy that jumps above the sustained level as kick; steady bass notes survive.
    rows, _ = band_rows(0, BAND_EDGES_HZ[0] * fade, 0)
    held = sustained(mag[rows], BASS["sustain_s"])
    burst = mag[rows] - held
    drum[rows] = burst**2 / (burst**2 + (scale * BASS["margin"] * held) ** 2 + eps)

    if progress:
        progress("Finding drums", 0.35)
    held = sustained(mag, MID["sustain_s"])  # MID and TREBLE share the same sustain window
    bands = ((MID, BAND_EDGES_HZ[0] / fade, BAND_EDGES_HZ[1] * fade, BAND_EDGES_HZ[0]),
             (TREBLE, BAND_EDGES_HZ[1] / fade, freqs[-1], BAND_EDGES_HZ[1]))
    for cfg, lo_hz, hi_hz, edge in bands:
        k = _odd(cfg["spread_hz"] * n_fft / SR)
        padded, inner = band_rows(lo_hz, hi_hz, k)
        block = mag[padded]
        broadband = np.minimum(median_filter(block, size=(k, 1), mode="reflect"), block)[inner]
        rows = slice(padded.start + inner.start, padded.start + inner.stop)
        mask = broadband**2 / (broadband**2 + (scale * cfg["margin"] * held[rows]) ** 2 + eps)
        # Cross-fade from the band below, centred on the edge between them.
        w = np.clip((np.log(np.maximum(freqs[rows], 1.0) / edge) / CROSSFADE + 1) / 2, 0, 1)[:, None]
        drum[rows] = drum[rows] * (1 - w) + mask * w

    keep = (1.0 - drum).astype(np.float32)

    if progress:
        progress("Rebuilding audio", 0.7)
    out = np.stack([librosa.istft(S * keep, hop_length=hop, n_fft=n_fft, length=n) for S in specs])
    return out.astype(np.float32)


# ---------------------------------------------------------------- Demucs (local AI)

def demucs_available() -> bool:
    try:
        import demucs  # noqa: F401
        return True
    except ImportError:
        return False


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


def remove_drums_demucs(audio: np.ndarray, progress: Progress | None = None) -> np.ndarray:
    """Run htdemucs in-process and return everything except the drum stem.

    Calls the Demucs library directly (not its CLI) so we never touch torchaudio's
    file I/O, which newer torchaudio releases removed. The model (~80 MB) downloads
    once on first use and is cached by torch, then stays loaded between songs.
    """
    if not demucs_available():
        raise RuntimeError("Demucs isn't installed. Run: ./start.command --with-ai")
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
