"""Drumless — drop audio files in, get drum-free copies out.

Run:  python app.py      then open http://127.0.0.1:8765
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

import separation

HERE = Path(__file__).parent
CONFIG_DIR = Path.home() / ".drumless"
CONFIG_FILE = CONFIG_DIR / "config.json"
MANIFEST_NAME = ".drumless-manifest.json"
TMP_DIR = Path(tempfile.gettempdir()) / "drumless-uploads"
TMP_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_EXTS = {".mp3", ".m4a", ".aac", ".wav", ".aif", ".aiff", ".flac", ".ogg", ".opus", ".alac", ".caf", ".wma"}
DEFAULTS = {
    "save_dir": str(Path.home() / "Music" / "Drumless"),
    "method": "dsp",        # "dsp" (no AI) or "demucs"
    "strength": "normal",   # dsp only: gentle / normal / aggressive
    "format": "auto",       # auto (same as the source file) / m4a / flac / wav / mp3
    "rename_title": True,   # append " (No Drums)" to the title tag
}

app = FastAPI(title="Drumless")
lock = threading.Lock()
jobs: dict[str, dict] = {}
work: "queue.Queue[str]" = queue.Queue()


# ---------------------------------------------------------------- config

def load_config() -> dict:
    try:
        return {**DEFAULTS, **json.loads(CONFIG_FILE.read_text())}
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(DEFAULTS)


def save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


# ---------------------------------------------------------------- manifest (dedupe)
# Stored inside the save folder so it travels with the output. Keyed by the
# SHA-256 of the *source file*, so renaming or moving a song doesn't fool it.

def manifest_path(save_dir: Path) -> Path:
    return save_dir / MANIFEST_NAME


def load_manifest(save_dir: Path) -> dict:
    try:
        return json.loads(manifest_path(save_dir).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"sources": {}, "outputs": {}}


def write_manifest(save_dir: Path, data: dict) -> None:
    tmp = manifest_path(save_dir).with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(manifest_path(save_dir))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- naming

def clean(part: str, fallback: str) -> str:
    part = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", (part or "").strip()).strip(". ")
    return part[:120] or fallback


def first(tags: dict, *keys: str) -> str:
    """First non-empty tag. iTunes Store files often carry only sort_* fields,
    because the Music app keeps the display names in its own library database."""
    return next((tags[k] for k in keys if tags.get(k)), "")


def output_path(save_dir: Path, source_name: str, tags: dict, fmt: str) -> Path:
    """Save Dir / Artist / Album / <original name> (No Drums).<fmt> — mirrors a music library,
    so two different "01 Intro.m4a" files from different albums don't collide."""
    artist = clean(first(tags, "album_artist", "albumartist", "artist", "sort_album_artist", "sort_artist"), "Unknown Artist")
    album = clean(first(tags, "album", "sort_album"), "Unknown Album")
    stem = clean(Path(source_name).stem, "Track")
    return save_dir / artist / album / f"{stem} (No Drums).{fmt}"


# ---------------------------------------------------------------- jobs

def update(job_id: str, **fields) -> None:
    with lock:
        jobs[job_id].update(fields)


def worker() -> None:
    while True:
        job_id = work.get()
        job = jobs[job_id]
        try:
            process(job_id, job)
        except Exception as e:  # surface any failure to the UI
            update(job_id, status="error", stage="Failed", error=str(e), progress=1.0)
        finally:
            Path(job["tmp_path"]).unlink(missing_ok=True)
            work.task_done()


def process(job_id: str, job: dict) -> None:
    cfg = job["config"]
    src = Path(job["tmp_path"])
    save_dir = Path(cfg["save_dir"])
    dest = Path(job["output"])

    def progress(stage: str, frac: float) -> None:
        update(job_id, stage=stage, progress=round(frac, 3))

    update(job_id, status="working", started=time.time())
    progress("Decoding", 0.05)

    if cfg["method"] == "demucs":
        out = separation.remove_drums_demucs(separation.decode(src), progress)
    else:
        out = separation.remove_drums_dsp(separation.decode(src), cfg["strength"], progress)

    progress("Saving", 0.9)
    dest.parent.mkdir(parents=True, exist_ok=True)
    title = first(job["tags"], "title", "sort_name") or Path(job["name"]).stem
    partial = dest.with_name(dest.stem + ".partial" + dest.suffix)
    separation.encode(out, src, partial, cfg["format"], f"{title} (No Drums)" if cfg["rename_title"] else None)
    partial.replace(dest)

    with lock:  # manifest writes are serialised with the dedupe check in /api/upload
        m = load_manifest(save_dir)
        m["sources"][job["hash"]] = {
            "source": job["name"], "output": str(dest.relative_to(save_dir)),
            "method": cfg["method"], "date": datetime.now().isoformat(timespec="seconds"),
        }
        m["outputs"][sha256_file(dest)] = str(dest.relative_to(save_dir))
        write_manifest(save_dir, m)
        jobs[job_id].update(status="done", stage="Done", progress=1.0,
                            seconds=round(time.time() - job["started"], 1))


threading.Thread(target=worker, daemon=True).start()


# ---------------------------------------------------------------- API

class ConfigIn(BaseModel):
    save_dir: Optional[str] = None
    method: Optional[str] = None
    strength: Optional[str] = None
    format: Optional[str] = None
    rename_title: Optional[bool] = None


@app.get("/api/config")
def get_config():
    cfg = load_config()
    return {**cfg, "demucs_available": separation.demucs_available(),
            "can_pick_folder": platform.system() == "Darwin"}


@app.post("/api/config")
def set_config(body: ConfigIn):
    cfg = load_config()
    changes = body.model_dump(exclude_none=True)
    if "save_dir" in changes:
        p = Path(changes["save_dir"]).expanduser()
        if not p.is_absolute():
            raise HTTPException(400, f"Use a full folder path, like {Path.home() / 'Music' / 'Drumless'}")
        changes["save_dir"] = str(p)
    if changes.get("method") not in (None, "dsp", "demucs"):
        raise HTTPException(400, "Unknown method")
    if changes.get("method") == "demucs" and not separation.demucs_available():
        raise HTTPException(400, "Demucs isn't installed yet. Run: pip install demucs")
    if changes.get("format") not in (None, "auto", *separation.CODECS):
        raise HTTPException(400, "Unknown format")
    if changes.get("strength") not in (None, *separation.STRENGTH_MARGINS):
        raise HTTPException(400, "Unknown strength")
    cfg.update(changes)
    save_config(cfg)
    return get_config()


@app.post("/api/pick-folder")
def pick_folder():
    """Open the native macOS folder chooser (it appears on the Mac running the server)."""
    if platform.system() != "Darwin":
        raise HTTPException(400, "Folder picker only works on macOS. Type a path instead.")
    script = 'POSIX path of (choose folder with prompt "Where should Drumless save tracks?")'
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if r.returncode != 0:
        return {"cancelled": True}
    return set_config(ConfigIn(save_dir=r.stdout.strip().rstrip("/") or "/"))


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = Path(file.filename or "track").name
    if Path(name).suffix.lower() not in AUDIO_EXTS:
        raise HTTPException(400, f"{name} isn't an audio file I recognise.")

    cfg = load_config()
    cfg["format"] = separation.resolve_format(cfg["format"], name)
    save_dir = Path(cfg["save_dir"])
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise HTTPException(400, f"Can't use save folder {save_dir}: {e.strerror}")

    # Stream to disk while hashing, so big files never sit in memory.
    job_id = uuid.uuid4().hex[:12]
    tmp = TMP_DIR / f"{job_id}{Path(name).suffix.lower()}"
    h = hashlib.sha256()
    with open(tmp, "wb") as out:
        while chunk := await file.read(1 << 20):
            h.update(chunk)
            out.write(chunk)
    digest = h.hexdigest()
    tags = separation.probe_tags(tmp)

    job = {"id": job_id, "name": name, "hash": digest, "tags": tags, "config": cfg,
           "tmp_path": str(tmp), "status": "queued", "stage": "Waiting", "progress": 0.0,
           "created": time.time()}

    with lock:
        manifest = load_manifest(save_dir)
        prior = manifest["sources"].get(digest)
        busy = next((j for j in jobs.values() if j["hash"] == digest
                     and j["status"] in ("queued", "working") and j["config"]["save_dir"] == str(save_dir)), None)
        dest = output_path(save_dir, name, tags, cfg["format"])

        reason = None
        if digest in manifest["outputs"]:
            reason = "This is already a drumless file."
        elif prior and (save_dir / prior["output"]).exists():
            dest, reason = save_dir / prior["output"], "Already done — skipped."
        elif busy:
            reason = "Already in the queue."
        elif dest.exists():
            reason = "A drumless copy already exists — skipped."

        job["output"] = str(dest)
        if reason:
            job.update(status="skipped", stage=reason, progress=1.0)
            tmp.unlink(missing_ok=True)
        jobs[job_id] = job

    if not reason:
        work.put(job_id)
    return public(job)


def public(job: dict) -> dict:
    keys = ("id", "name", "status", "stage", "progress", "error", "seconds", "output", "created")
    out = {k: job.get(k) for k in keys}
    out["title"] = first(job["tags"], "title", "sort_name")
    out["artist"] = first(job["tags"], "artist", "album_artist", "sort_artist")
    out["method"] = job["config"]["method"]
    out["has_output"] = job["status"] in ("done", "skipped") and Path(job["output"]).exists()
    return out


@app.get("/api/jobs")
def list_jobs():
    with lock:
        return sorted((public(j) for j in jobs.values()), key=lambda j: j["created"], reverse=True)


@app.post("/api/jobs/clear")
def clear_finished():
    with lock:
        for k in [k for k, j in jobs.items() if j["status"] in ("done", "skipped", "error")]:
            del jobs[k]
    return {"ok": True}


@app.get("/api/jobs/{job_id}/audio")
def job_audio(job_id: str):
    job = jobs.get(job_id)
    if not job or not Path(job["output"]).exists():
        raise HTTPException(404)
    return FileResponse(job["output"])


@app.post("/api/jobs/{job_id}/reveal")
def reveal(job_id: str):
    job = jobs.get(job_id)
    if not job or not Path(job["output"]).exists():
        raise HTTPException(404)
    if platform.system() == "Darwin":
        subprocess.run(["open", "-R", job["output"]])
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


if __name__ == "__main__":
    host = os.environ.get("DRUMLESS_HOST", "127.0.0.1")
    port = int(os.environ.get("DRUMLESS_PORT", "8765"))
    if not shutil.which("ffmpeg"):
        print(f"⚠️  ffmpeg is missing. Install it with: {separation.FFMPEG_INSTALL}")
    print(f"Drumless running at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
