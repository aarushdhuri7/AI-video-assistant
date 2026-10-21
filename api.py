"""FastAPI backend.

Run:  python api.py
  or: python -m uvicorn api:app --host 127.0.0.1 --port 8000

Analyses are kept IN MEMORY: they (and their chat indexes) are lost when the server restarts.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core import PipelineError
from core.llm_factory import check_ollama
from core.rag_engine import ask_question
from core.transcriber import LANGUAGES
from main import run_pipeline
from utils.audio_processor import ALLOWED_EXTENSIONS, ffmpeg_path, safe_display_name, validate_extension

load_dotenv()
log = logging.getLogger("video-assistant")
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "uploads"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
MAX_ANALYSES = int(os.getenv("MAX_ANALYSES", "20"))  # oldest finished ones are dropped beyond this
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]

app = FastAPI(title="AI Video Assistant")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

# One analysis at a time: Whisper and the LLM are heavy.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis")
_lock = threading.Lock()
_analyses: dict[str, dict] = {}  # id -> record (the RAG chain lives inside, never serialised)


# ------------------------------------------------------------------ schemas
class AnalysisSummary(BaseModel):
    id: str
    status: str  # pending | processing | completed | error
    filename: str
    language: str
    created_at: str
    title: Optional[str] = None


class AnalysisOut(AnalysisSummary):
    stage: Optional[str] = None
    error: Optional[str] = None
    transcript: Optional[str] = None
    summary: Optional[str] = None
    action_items: Optional[str] = None
    key_decisions: Optional[str] = None
    open_questions: Optional[str] = None


class ChatIn(BaseModel):
    question: str = Field(..., max_length=2000)


class ChatOut(BaseModel):
    answer: str


# ----------------------------------------------------------------- errors
@app.exception_handler(RequestValidationError)
async def _validation_error(_, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    loc = ".".join(str(p) for p in first.get("loc", []) if p not in ("body", "query"))
    return JSONResponse(status_code=422, content={"detail": f"{loc}: {first.get('msg', 'invalid request')}" if loc else "Invalid request."})


@app.exception_handler(Exception)
async def _unhandled(_, exc: Exception):
    log.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Something went wrong on the server."})


# ---------------------------------------------------------------- helpers
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _public(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k != "rag_chain"}


def _get(analysis_id: str) -> dict:
    with _lock:
        rec = _analyses.get(analysis_id)
    if rec is None:
        raise HTTPException(404, "Analysis not found. It may have been lost when the server restarted.")
    return rec


def check_environment() -> Optional[str]:
    """Return a user-facing problem description, or None if FFmpeg and Ollama are ready."""
    if ffmpeg_path() is None:
        return "FFmpeg isn't installed (or isn't on PATH). Install it, then restart the terminal and the server."
    return check_ollama()


def _evict_old() -> None:
    with _lock:
        finished = [r for r in _analyses.values() if r["status"] in ("completed", "error")]
        excess = len(_analyses) - MAX_ANALYSES
        for rec in sorted(finished, key=lambda r: r["created_at"])[: max(excess, 0)]:
            _analyses.pop(rec["id"], None)
            chain = rec.get("rag_chain")
            if chain is not None:
                chain.close()


def _run_job(analysis_id: str, path: Path, language: str) -> None:
    rec = _analyses[analysis_id]

    def progress(stage: str) -> None:
        rec["stage"] = stage

    rec["status"] = "processing"
    try:
        result = run_pipeline(path, language, progress=progress)
        rec.update({k: result[k] for k in ("title", "transcript", "summary", "action_items", "key_decisions", "open_questions")})
        rec["rag_chain"] = result["rag_chain"]
        rec["status"] = "completed"
    except PipelineError as exc:
        rec["status"], rec["error"] = "error", str(exc)
    except Exception:  # noqa: BLE001
        log.exception("Analysis %s failed", analysis_id)
        rec["status"], rec["error"] = "error", "The analysis failed unexpectedly. Check the server log for details."
    finally:
        rec["stage"] = None
        shutil.rmtree(path.parent, ignore_errors=True)  # delete the uploaded copy
        _evict_old()


# ----------------------------------------------------------------- routes
@app.get("/")
def health() -> dict:
    return {"status": "ok", "service": "AI Video Assistant"}


@app.get("/api/analyses", response_model=list[AnalysisSummary])
def list_analyses() -> list[dict]:
    with _lock:
        items = sorted(_analyses.values(), key=lambda r: r["created_at"], reverse=True)
        return [_public(r) for r in items]


@app.post("/api/analyses/upload", response_model=AnalysisOut, status_code=202)
def upload(file: UploadFile = File(...), language: str = Form("english")) -> dict:
    language = language.lower()
    if language not in LANGUAGES:
        raise HTTPException(422, f"Unsupported language. Choose one of: {', '.join(LANGUAGES)}.")
    display = safe_display_name(file.filename or "")
    try:
        ext = validate_extension(display)
    except PipelineError as exc:
        raise HTTPException(415, str(exc))
    problem = check_environment()
    if problem:
        raise HTTPException(503, problem)

    analysis_id = str(uuid.uuid4())
    folder = UPLOAD_DIR / analysis_id
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"input{ext}"  # server-chosen name: the client's filename is never used as a path
    limit, written = MAX_UPLOAD_MB * 1024 * 1024, 0
    try:
        with open(dest, "wb") as out:
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(413, f"File is too large. The limit is {MAX_UPLOAD_MB} MB.")
                out.write(chunk)
        if written == 0:
            raise HTTPException(422, "The uploaded file is empty.")
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise

    rec = {
        "id": analysis_id, "status": "pending", "stage": None, "filename": display, "language": language,
        "created_at": _now(), "title": None, "error": None, "transcript": None, "summary": None,
        "action_items": None, "key_decisions": None, "open_questions": None, "rag_chain": None,
    }
    with _lock:
        _analyses[analysis_id] = rec
    snapshot = _public(rec)  # taken before the worker can change the status
    _executor.submit(_run_job, analysis_id, dest, language)
    return snapshot


@app.get("/api/analyses/{analysis_id}", response_model=AnalysisOut)
def get_analysis(analysis_id: str) -> dict:
    return _public(_get(analysis_id))


@app.post("/api/analyses/{analysis_id}/chat", response_model=ChatOut)
def chat(analysis_id: str, body: ChatIn) -> dict:
    rec = _get(analysis_id)
    question = body.question.strip()
    if not question:
        raise HTTPException(422, "Please enter a question.")
    if rec["status"] != "completed" or rec.get("rag_chain") is None:
        raise HTTPException(409, "This analysis isn't finished yet." if rec["status"] in ("pending", "processing") else "Chat isn't available for this analysis.")
    try:
        return {"answer": ask_question(rec["rag_chain"], question)}
    except PipelineError as exc:
        raise HTTPException(502, str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
