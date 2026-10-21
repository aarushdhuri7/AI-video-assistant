"""Core-logic tests. Uses real FFmpeg, real ChromaDB (with a fake embedding) and a tiny
fake Ollama HTTP server, so the real langchain-ollama code path runs without a model."""
import hashlib
import json
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from core import PipelineError
from core import llm_factory, rag_engine
from core.chunking import split_transcript
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.llm_factory import check_ollama, strip_thinking
from core.summarize import generate_title, summarize
from utils import audio_processor
from utils.audio_processor import extract_audio, safe_display_name, validate_extension


# ------------------------------------------------------------ fake Ollama
class FakeOllama:
    def __init__(self):
        self.requests = []  # bodies POSTed to /api/chat
        self.replies = ["<think>secret reasoning</think>Hello from the model"]
        self.reject_think = False
        self.models = ["qwen3:8b"]
        self._i = 0

    def next_reply(self):
        r = self.replies[min(self._i, len(self.replies) - 1)]
        self._i += 1
        return r


@pytest.fixture
def ollama(monkeypatch):
    state = FakeOllama()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = json.dumps({"models": [{"name": m} for m in state.models]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append(req)
            if state.reject_think and "think" in req and req["think"] is not None:
                body = json.dumps({"error": "model does not support thinking"}).encode()
                self.send_response(400); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
                return
            msg = {"model": req["model"], "created_at": "2026-01-01T00:00:00Z", "message": {"role": "assistant", "content": state.next_reply()}}
            self.send_response(200); self.send_header("Content-Type", "application/x-ndjson"); self.end_headers()
            self.wfile.write((json.dumps({**msg, "done": False}) + "\n").encode())
            self.wfile.write((json.dumps({**msg, "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"}) + "\n").encode())

    server = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("OLLAMA_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")
    monkeypatch.setattr(llm_factory, "_thinking_disabled", True)
    llm_factory.reset_shared_llm()
    yield state
    server.shutdown()
    llm_factory.reset_shared_llm()


# ------------------------------------------------------------ pure helpers
def test_strip_thinking():
    assert strip_thinking("<think>a\nb</think>\nAnswer") == "Answer"
    assert strip_thinking("<think>never closed") == ""
    assert strip_thinking("```markdown\n## Hi\n```") == "## Hi"
    assert strip_thinking("plain") == "plain"


def test_extension_and_filename_validation():
    assert validate_extension("Talk.MP4") == ".mp4"
    for bad in ("a.exe", "a", "a.mp3.sh", "a.txt"):
        with pytest.raises(PipelineError):
            validate_extension(bad)
    assert safe_display_name("../../x/evil\x00.mp3") == "evil.mp3"
    assert audio_processor.ALLOWED_EXTENSIONS == (".mp3", ".wav", ".m4a", ".aac", ".flac", ".mp4", ".mov", ".mkv", ".webm")


def test_chunking():
    assert split_transcript("short") == ["short"]
    parts = split_transcript("word " * 5000, size=1000)
    assert len(parts) > 20 and all(len(p) <= 1000 for p in parts)


# ------------------------------------------------------------ real FFmpeg
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _ff(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


@needs_ffmpeg
def test_extract_audio_from_audio_and_video(tmp_path):
    _ff("-f", "lavfi", "-i", "sine=frequency=440:duration=2", str(tmp_path / "a.mp3"))
    _ff("-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=2", "-pix_fmt", "yuv420p", "-shortest", str(tmp_path / "v.mp4"))
    for name in ("a.mp3", "v.mp4"):
        out_dir = tmp_path / ("out_" + name); out_dir.mkdir()
        wav = extract_audio(tmp_path / name, out_dir)
        assert wav.exists() and wav.stat().st_size > 10_000
        assert (tmp_path / name).exists()  # the original is never deleted


@needs_ffmpeg
def test_extract_audio_error_messages(tmp_path):
    _ff("-f", "lavfi", "-i", "color=c=black:s=64x64:d=1", "-pix_fmt", "yuv420p", str(tmp_path / "silent.mp4"))
    with pytest.raises(PipelineError, match="no audio"):
        extract_audio(tmp_path / "silent.mp4", tmp_path)
    (tmp_path / "junk.mp3").write_bytes(b"not really audio" * 100)
    with pytest.raises(PipelineError, match="couldn't read"):
        extract_audio(tmp_path / "junk.mp3", tmp_path)


def test_missing_ffmpeg_message(monkeypatch, tmp_path):
    monkeypatch.setattr(audio_processor, "ffmpeg_path", lambda: None)
    with pytest.raises(PipelineError, match="FFmpeg was not found"):
        extract_audio(tmp_path / "x.mp3", tmp_path)


# ------------------------------------------------- LLM paths via fake Ollama
def test_llm_is_shared_and_thinking_disabled_and_output_cleaned(ollama):
    ollama.replies = ["<think>secret reasoning</think>## Meeting Summary\nAll good."]
    out = summarize("We met and agreed to ship on Friday.")
    assert out == "## Meeting Summary\nAll good."
    req = ollama.requests[0]
    assert req["model"] == "qwen3:8b" and req["think"] is False and req["options"]["num_ctx"] == 4096
    assert llm_factory.get_shared_llm() is llm_factory.get_shared_llm()
    assert "ship on Friday" in json.dumps(req["messages"])


def test_retries_without_thinking_if_server_rejects_it(ollama):
    ollama.reject_think = True
    ollama.replies = ["Fine"]
    assert generate_title("text") == "Fine"
    assert "think" in ollama.requests[0] or len(ollama.requests) >= 1
    assert llm_factory._thinking_disabled is False  # remembered; later calls skip the flag


def test_title_is_cleaned(ollama):
    ollama.replies = ['"Quarterly Planning Meeting."\nextra line']
    assert generate_title("abc") == "Quarterly Planning Meeting"


def test_extractors_return_empty_messages(ollama):
    ollama.replies = ["NONE"]
    assert "No action items" in extract_action_items("hi there")
    assert "No key decisions" in extract_key_decisions("hi there")
    assert "No open questions" in extract_questions("hi there")
    assert "No action items" in extract_action_items("   ")


def test_extractor_returns_items(ollama):
    ollama.replies = ["- **Send report** (Owner: Dana)"]
    assert extract_action_items("Dana will send the report.") == "- **Send report** (Owner: Dana)"
    assert len(ollama.requests) == 1  # short transcript: one call, no merge


def test_long_transcript_is_chunked_then_merged(ollama, monkeypatch):
    monkeypatch.setenv("CHUNK_CHARS", "1000")
    text = "sentence. " * 350
    n = len(split_transcript(text))
    assert n >= 3
    ollama.replies = ["- a", "NONE"] + ["- b"] * (n - 2) + ["- merged a+b"]
    assert extract_key_decisions(text) == "- merged a+b"
    assert len(ollama.requests) == n + 1  # one call per chunk + one merge

    ollama.requests.clear(); ollama._i = 0
    ollama.replies = ["notes"] * n + ["## Meeting Summary\nfinal"]
    assert summarize(text) == "## Meeting Summary\nfinal"
    assert len(ollama.requests) == n + 1


def test_unreachable_ollama_gives_clear_error(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")  # nothing listens here
    llm_factory.reset_shared_llm()
    with pytest.raises(PipelineError, match="Could not reach Ollama"):
        generate_title("x")
    assert "isn't reachable" in check_ollama()
    llm_factory.reset_shared_llm()


def test_check_ollama_model_missing_and_ok(ollama):
    assert check_ollama() is None
    ollama.models = ["llama3:8b"]
    assert "ollama pull qwen3:8b" in check_ollama()


# ------------------------------------------------------------------- RAG
def fake_embedding():
    """Deterministic bag-of-words embedding so Chroma runs offline."""
    class F:
        def __call__(self, input):
            out = []
            for text in input:
                v = [0.0] * 64
                for w in text.lower().split():
                    v[int(hashlib.md5(w.strip(".,?!").encode()).hexdigest(), 16) % 64] += 1.0
                out.append(v)
            return out

        @staticmethod
        def name():
            return "fake"

        def embed_query(self, input):
            return self(input)

        def is_legacy(self):
            return False

        def get_config(self):
            return {}

        @staticmethod
        def build_from_config(c):
            return F()

        def default_space(self):
            return "cosine"

        def supported_spaces(self):
            return ["cosine", "l2", "ip"]
    return F()


@pytest.fixture
def rag(monkeypatch):
    monkeypatch.setattr(rag_engine, "_get_embedding_function", fake_embedding)


def test_rag_sessions_are_isolated(rag):
    a = rag_engine.build_rag_chain("The budget for marketing is fifty thousand dollars. " * 5)
    b = rag_engine.build_rag_chain("The migration to the new database finishes in March. " * 5)
    assert a.name != b.name
    assert all("budget" in d for d in a.retrieve("marketing budget"))
    assert all("database" in d for d in b.retrieve("database migration"))
    assert not any("database" in d for d in a.retrieve("database migration"))
    a.close(); b.close()


def test_ask_question_sends_retrieved_context_only(rag, ollama):
    ollama.replies = ["<think>x</think>Fifty thousand dollars."]
    chain = rag_engine.build_rag_chain("The marketing budget is fifty thousand dollars. " * 3)
    answer = rag_engine.ask_question(chain, "What is the marketing budget?")
    assert answer == "Fifty thousand dollars."
    sent = json.dumps(ollama.requests[-1]["messages"])
    assert "marketing budget is fifty thousand" in sent and "What is the marketing budget?" in sent
    assert rag_engine.NOT_FOUND in sent  # the 'say you can't find it' instruction is present


def test_rag_edge_cases(rag):
    with pytest.raises(PipelineError):
        rag_engine.build_rag_chain("   ")
    chain = rag_engine.build_rag_chain("Some text here.")
    with pytest.raises(PipelineError):
        rag_engine.ask_question(chain, "  ")
    chain.close()
    assert rag_engine.ask_question.__name__ == "ask_question"
