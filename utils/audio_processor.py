"""Validate uploads and turn any supported audio/video file into 16 kHz mono WAV.

Uses the FFmpeg command line directly (arguments are passed as a list, never
through a shell), so no extra Python audio library is needed.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from core import PipelineError

ALLOWED_EXTENSIONS = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".mp4", ".mov", ".mkv", ".webm")


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def safe_display_name(filename: str) -> str:
    """Basename only, with control characters removed."""
    name = (filename or "").replace("\\", "/").split("/")[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()
    return name[:120] or "upload"


def validate_extension(filename: str) -> str:
    ext = Path(safe_display_name(filename)).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise PipelineError(
            "Unsupported file type. Use one of: " + ", ".join(e[1:] for e in ALLOWED_EXTENSIONS) + "."
        )
    return ext


def extract_audio(source: Path, out_dir: Path) -> Path:
    """Convert ``source`` to ``out_dir/audio.wav``. Never touches the source file."""
    exe = ffmpeg_path()
    if exe is None:
        raise PipelineError(
            "FFmpeg was not found. Install it and make sure `ffmpeg` works in a new terminal "
            "(Windows: winget install Gyan.FFmpeg)."
        )
    source = Path(source)
    if not source.is_file():
        raise PipelineError("The uploaded file could not be found.")
    out = Path(out_dir) / "audio.wav"
    cmd = [exe, "-y", "-nostdin", "-loglevel", "error", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(out)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired as exc:
        raise PipelineError("Converting the file took too long and was stopped.") from exc
    if proc.returncode != 0 or not out.exists() or out.stat().st_size < 1000:
        err = (proc.stderr or "").lower()
        if "does not contain any stream" in err or "output file is empty" in err or (proc.returncode == 0 and out.exists()):
            raise PipelineError("This file has no audio track.")
        raise PipelineError("FFmpeg couldn't read this file. It may be corrupt or in an unsupported format.")
    return out
