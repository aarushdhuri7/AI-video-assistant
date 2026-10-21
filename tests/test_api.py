"""API tests. The AI pipeline is replaced by a fake; no Ollama, Whisper or FFmpeg needed."""
import io
import time

import pytest
from fastapi.testclient import TestClient

import api
from core import PipelineError


class FakeChain:
    def __init__(self, tag):
        self.tag = tag
        self.closed = False

    def close(self):
        self.closed = True


def fake_pipeline(path, language, progress=None):
    progress("Transcribing")
    assert path.is_file()  # the upload really exists while the job runs
    return {
        "title": "Weekly Sync", "transcript": f"hello [{language}]", "summary": "## Meeting Summary\nok",
        "action_items": "- **Ship it**", "key_decisions": "- Go", "open_questions": "No open questions were found in this transcript.",
        "rag_chain": FakeChain(path.parent.name),
    }


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(api, "check_environment", lambda: None)
    monkeypatch.setattr(api, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(api, "ask_question", lambda chain, q: f"answer to '{q}' from {chain.tag}")
    api._analyses.clear()
    yield tmp_path / "uploads"


@pytest.fixture
def client():
    return TestClient(api.app)


def upload(client, name="talk.mp3", content=b"abc", language="english"):
    return client.post("/api/analyses/upload", files={"file": (name, io.BytesIO(content), "application/octet-stream")}, data={"language": language})


def wait(client, aid, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        d = client.get(f"/api/analyses/{aid}").json()
        if d["status"] in ("completed", "error"):
            return d
        time.sleep(0.02)
    raise AssertionError("timeout")


def test_health(client):
    assert client.get("/").json()["status"] == "ok"


def test_no_youtube_endpoint(client):
    assert client.post("/api/analyses", json={"url": "https://youtu.be/x"}).status_code in (404, 405)


def test_upload_flow_and_results(client, isolated):
    r = upload(client, language="Hindi")
    assert r.status_code == 202 and r.json()["status"] == "pending"
    d = wait(client, r.json()["id"])
    assert d["status"] == "completed" and d["title"] == "Weekly Sync"
    assert d["transcript"] == "hello [hindi]" and d["filename"] == "talk.mp3"
    assert "rag_chain" not in d
    assert list(isolated.iterdir()) == []  # uploaded copy deleted


def test_list(client):
    assert client.get("/api/analyses").json() == []
    aid = upload(client).json()["id"]
    wait(client, aid)
    assert [a["id"] for a in client.get("/api/analyses").json()] == [aid]
    assert "transcript" not in client.get("/api/analyses").json()[0]


@pytest.mark.parametrize("name", ["a.exe", "a.txt", "noext", "a.mp3.sh"])
def test_rejects_bad_extension(client, name):
    assert upload(client, name=name).status_code == 415


def test_rejects_bad_language_empty_and_large(client, monkeypatch, isolated):
    assert upload(client, language="klingon").status_code == 422
    assert upload(client, content=b"").status_code == 422
    monkeypatch.setattr(api, "MAX_UPLOAD_MB", 1)
    assert upload(client, content=b"x" * (2 * 1024 * 1024)).status_code == 413
    assert list(isolated.iterdir()) == []


def test_filename_cannot_escape(client, isolated):
    r = upload(client, name="../../evil.mp3")
    assert r.status_code == 202 and r.json()["filename"] == "evil.mp3"
    wait(client, r.json()["id"])


def test_environment_problem_returns_503(client, monkeypatch):
    monkeypatch.setattr(api, "check_environment", lambda: "Ollama isn't reachable")
    r = upload(client)
    assert r.status_code == 503 and "Ollama" in r.json()["detail"]
    assert api._analyses == {}


def test_pipeline_error_is_shown_but_crash_details_are_not(client, monkeypatch):
    def bad(path, language, progress=None):
        raise PipelineError("No speech was detected in this file.")

    monkeypatch.setattr(api, "run_pipeline", bad)
    d = wait(client, upload(client).json()["id"])
    assert d["status"] == "error" and d["error"] == "No speech was detected in this file."

    def crash(path, language, progress=None):
        raise RuntimeError("secret /home/x/key=123")

    monkeypatch.setattr(api, "run_pipeline", crash)
    d = wait(client, upload(client).json()["id"])
    assert d["status"] == "error" and "secret" not in str(d)


def test_chat_uses_the_right_session(client):
    a, b = upload(client).json()["id"], upload(client).json()["id"]
    wait(client, a), wait(client, b)
    ra = client.post(f"/api/analyses/{a}/chat", json={"question": "hi"}).json()["answer"]
    rb = client.post(f"/api/analyses/{b}/chat", json={"question": "hi"}).json()["answer"]
    assert ra.endswith(a) and rb.endswith(b) and ra != rb


def test_chat_errors(client, monkeypatch):
    assert client.post("/api/analyses/nope/chat", json={"question": "hi"}).status_code == 404
    monkeypatch.setattr(api, "run_pipeline", lambda p, l, progress=None: (time.sleep(0.3), fake_pipeline(p, l, progress))[1])
    aid = upload(client).json()["id"]
    assert client.post(f"/api/analyses/{aid}/chat", json={"question": "hi"}).status_code == 409
    wait(client, aid)
    assert client.post(f"/api/analyses/{aid}/chat", json={"question": "   "}).status_code == 422
    assert client.post(f"/api/analyses/{aid}/chat", json={}).status_code == 422


def test_unknown_analysis_404(client):
    assert client.get("/api/analyses/does-not-exist").status_code == 404


def test_old_analyses_are_evicted_and_their_index_closed(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_ANALYSES", 2)
    ids = []
    for _ in range(3):
        ids.append(upload(client).json()["id"])
        wait(client, ids[-1])
    deadline = time.time() + 2
    while len(api._analyses) > 2 and time.time() < deadline:
        time.sleep(0.02)
    assert ids[0] not in api._analyses and len(api._analyses) == 2


def test_cors_only_for_configured_origins(client):
    ok = client.get("/", headers={"Origin": "http://localhost:5173"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "access-control-allow-origin" not in client.get("/", headers={"Origin": "http://evil.example"}).headers
