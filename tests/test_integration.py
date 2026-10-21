"""End-to-end through the real code: api.py -> main.run_pipeline -> FFmpeg -> LangChain/Ollama HTTP -> Chroma.

Only two things are stubbed because their models can't be assumed to be present:
Whisper's model object and Chroma's embedding model.
"""
import glob
import io
import json
import shutil
import subprocess
import tempfile
import time

import pytest
from fastapi.testclient import TestClient

import api
from core import rag_engine, transcriber
from tests.test_core import fake_embedding, ollama  # noqa: F401  (fixtures)

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

SPOKEN = "Hello team. We decided to ship on Friday. Dana will send the report to finance. Who owns the budget?"


class FakeWhisper:
    class device:
        type = "cpu"

    seen = []

    def transcribe(self, path, **kw):
        import wave

        with wave.open(path) as w:  # proves FFmpeg produced a real 16 kHz mono WAV
            FakeWhisper.seen.append((w.getframerate(), w.getnchannels(), kw.get("language")))
        return {"text": f"  {SPOKEN}  "}


@pytest.fixture
def stack(monkeypatch, tmp_path, ollama):  # noqa: F811
    monkeypatch.setattr(transcriber, "_model", FakeWhisper())
    monkeypatch.setattr(rag_engine, "_get_embedding_function", fake_embedding)
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path / "uploads")
    api._analyses.clear()
    FakeWhisper.seen.clear()
    return TestClient(api.app)


def make_audio(tmp_path, name="meeting.mp3"):
    p = tmp_path / name
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=2", str(p)], check=True)
    return p.read_bytes()


def wait(client, aid):
    for _ in range(200):
        d = client.get(f"/api/analyses/{aid}").json()
        if d["status"] in ("completed", "error"):
            return d
        time.sleep(0.05)
    raise AssertionError("timeout")


def test_full_pipeline_through_the_api(stack, ollama, tmp_path):  # noqa: F811
    ollama.replies = [
        "<think>x</think>Weekly Launch Sync",                                  # title
        "## Meeting Summary\nThe team set a **Friday** ship date.",           # summary
        "- **Send report to finance** (Owner: Dana)",                          # actions
        "- Ship on **Friday**",                                                # decisions
        "- Who owns the budget?",                                              # questions
        "The budget owner was not named.",                                     # chat answer
    ]
    audio = make_audio(tmp_path)
    r = stack.post("/api/analyses/upload", files={"file": ("meeting.mp3", io.BytesIO(audio), "audio/mpeg")}, data={"language": "english"})
    assert r.status_code == 202, r.text
    d = wait(stack, r.json()["id"])

    assert d["status"] == "completed", d
    assert d["title"] == "Weekly Launch Sync"
    assert d["transcript"] == SPOKEN
    assert d["summary"].startswith("## Meeting Summary") and "<think>" not in json.dumps(d)
    assert d["action_items"] == "- **Send report to finance** (Owner: Dana)"
    assert d["key_decisions"] == "- Ship on **Friday**" and d["open_questions"] == "- Who owns the budget?"
    assert FakeWhisper.seen == [(16000, 1, "en")]  # converted to 16 kHz mono; language hint passed

    # chat retrieves from THIS transcript and sends it to the model
    a = stack.post(f"/api/analyses/{d['id']}/chat", json={"question": "Who will send the report?"})
    assert a.status_code == 200 and a.json()["answer"] == "The budget owner was not named."
    sent = json.dumps(ollama.requests[-1]["messages"])
    assert "Dana will send the report" in sent and "Who will send the report?" in sent
    assert all(req["think"] is False for req in ollama.requests)  # thinking disabled on every call
    assert len(ollama.requests) == 6

    # temp files are gone and the user's original is untouched
    assert not (tmp_path / "uploads" / d["id"]).exists()
    assert glob.glob(f"{tempfile.gettempdir()}/vassist_*") == []


def test_hindi_language_hint_and_silent_video_error(stack, tmp_path):
    stack.post("/api/analyses/upload", files={"file": ("a.mp3", io.BytesIO(make_audio(tmp_path)), "audio/mpeg")}, data={"language": "hindi"})
    for _ in range(200):
        if FakeWhisper.seen:
            break
        time.sleep(0.05)
    assert FakeWhisper.seen[0][2] == "hi"

    silent = tmp_path / "silent.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1", "-pix_fmt", "yuv420p", str(silent)], check=True)
    r = stack.post("/api/analyses/upload", files={"file": ("silent.mp4", io.BytesIO(silent.read_bytes()), "video/mp4")}, data={"language": "english"})
    d = wait(stack, r.json()["id"])
    assert d["status"] == "error" and "no audio" in d["error"]


def test_ollama_down_is_reported_before_work_starts(stack, monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    r = stack.post("/api/analyses/upload", files={"file": ("a.mp3", io.BytesIO(b"abc"), "audio/mpeg")}, data={"language": "english"})
    assert r.status_code == 503 and "Ollama" in r.json()["detail"]
    assert api._analyses == {}
