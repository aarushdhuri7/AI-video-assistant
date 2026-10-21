# Video Assistant

Upload an audio or video file and get a transcript, a Markdown summary, action items, key decisions and open questions — then ask questions about it with RAG. Everything runs **locally**: Whisper for speech-to-text and Ollama (Qwen 3 8B) for the language model. No API keys are needed.

- **Backend:** Python 3.12, FastAPI, LangChain + `langchain-ollama`, local Whisper, ChromaDB, FFmpeg
- **Frontend:** React, Vite, TypeScript, Tailwind CSS, React Markdown (+ `remark-gfm`)
- Uploaded files only — there is no URL/YouTube processing.

```
ai-video-assistant/
├── api.py                  FastAPI app (run this)
├── main.py                 run_pipeline(): the orchestration, plus a tiny CLI
├── core/
│   ├── llm_factory.py      one shared Ollama chat model, <think> stripping, Ollama checks
│   ├── transcriber.py      cached local Whisper
│   ├── summarize.py        Markdown report + title
│   ├── extractor.py        action items / decisions / open questions
│   ├── rag_engine.py       ChromaDB index per analysis + ask_question()
│   └── chunking.py         splits long transcripts for the model
├── utils/audio_processor.py   upload validation + FFmpeg audio extraction
├── frontend/               React app
└── tests/                  pytest suite
```

---

## 1. Set up on Windows (PowerShell)

### 1.1 Install the prerequisites

| Tool | Install |
|---|---|
| Python 3.12 | https://www.python.org/downloads/ (tick **Add python.exe to PATH**) |
| Node.js 18+ | https://nodejs.org (LTS) |
| FFmpeg | `winget install Gyan.FFmpeg` — then **close and reopen PowerShell** and check `ffmpeg -version`. (Or download from https://www.gyan.dev/ffmpeg/builds/ and add its `bin` folder to PATH.) |
| Ollama | https://ollama.com/download, then `ollama pull qwen3:8b` (about 5 GB) |

Check that Ollama is running: `ollama list` should show `qwen3:8b`.

### 1.2 Backend

```powershell
cd ai-video-assistant
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# NVIDIA GPU (e.g. RTX 4050): install the CUDA build of PyTorch first.
# Pick the current command for your CUDA version at https://pytorch.org/get-started/locally/ , e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
copy .env.example .env                 # optional: edit settings
.\.venv\Scripts\python.exe api.py
```

(Skip the `torch` line to use the CPU. It works but is slower.) The API is now at http://127.0.0.1:8000 — open it and you should see `{"status":"ok", ...}`. The equivalent Uvicorn command is:

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

### 1.3 Frontend (a second terminal)

```powershell
cd ai-video-assistant\frontend
copy .env.example .env                 # VITE_API_BASE_URL=http://127.0.0.1:8000
npm install
npm run dev
```

Open http://localhost:5173.

### 1.4 First-run downloads

The first analysis downloads two small models, so it needs internet once:
- Whisper `small` (~460 MB) → `%USERPROFILE%\.cache\whisper`
- Chroma's embedding model, all-MiniLM-L6-v2 (~80 MB, runs on CPU) → `%USERPROFILE%\.cache\chroma`

After that everything works offline.

### 1.5 Run the tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

The tests need no Ollama, Whisper model or internet. (FFmpeg-based tests are skipped if FFmpeg isn't installed.)

---

## 2. Using it

1. **New Analysis** → drop in a file (mp3, wav, m4a, aac, flac, mp4, mov, mkv, webm), choose the spoken language (English, Hindi or Auto-detect), click **Analyze**.
2. The page shows the current stage while the backend works. Long recordings take minutes.
3. **Results & Insights** → Summary, Transcript, Action Items, Key Decisions, Open Questions.
4. **RAG Chat** → ask questions; answers come only from that recording's transcript, and the assistant says so when the answer isn't there.

Command line (optional): `python main.py path\to\file.mp4 english`

## 3. API

| Method & path | Purpose |
|---|---|
| `GET /` | health check |
| `GET /api/analyses` | list analyses (id, status, filename, title…) |
| `POST /api/analyses/upload` | multipart: `file`, `language` → `202` with the new analysis |
| `GET /api/analyses/{id}` | status (`pending`/`processing`/`completed`/`error`), current stage, and results |
| `POST /api/analyses/{id}/chat` | `{"question": "..."}` → `{"answer": "..."}` |

Analyses run one at a time in a background thread, so uploads return immediately. Before accepting an upload the API checks that FFmpeg exists and that Ollama is running with the model, and replies `503` with a plain-English message if not.

> **In-memory sessions:** analyses and their chat indexes live in the server's memory. They are **lost when the backend restarts** (the History list empties, and old links return "not found"). Only the 20 most recent are kept (`MAX_ANALYSES`).

## 4. Configuration (`.env`)

See `.env.example`; every value is optional.

| Variable | Default | Notes |
|---|---|---|
| `OLLAMA_MODEL` | `qwen3:8b` | any model you've pulled |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | |
| `OLLAMA_NUM_CTX` | `4096` | context size; raising it uses more VRAM |
| `OLLAMA_DISABLE_THINKING` | `true` | see below |
| `WHISPER_MODEL` | `small` | `tiny`/`base`/`small`/`medium`/`large` |
| `WHISPER_DEVICE` | auto | set `cpu` to keep the GPU free for the LLM |
| `CHUNK_CHARS` | `5000` | transcript piece size for summaries/extraction |
| `MAX_UPLOAD_MB` | `500` | |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | comma-separated |

**Thinking mode.** Qwen 3 normally writes a long hidden "thinking" trace. The app asks Ollama to turn it off (`reasoning=False` in `langchain-ollama`). If your Ollama/model rejects that option, the app retries once without it, and in every case any `<think>…</think>` text is stripped from the output.

**6 GB GPU tips.** Whisper `small` and Qwen3-8B together slightly exceed 6 GB; Ollama then places a few LLM layers on the CPU, which still works but is slower. Setting `WHISPER_DEVICE=cpu` or using `WHISPER_MODEL=base` frees memory. The app loads one LLM instance, serialises all LLM calls, and processes one analysis at a time. Nothing is loaded at startup.

## 5. How it behaves

- **Long transcripts** are split into ~5,000-character pieces; each piece is summarised/extracted, then the partial results are merged with one more call.
- **No invention:** prompts tell the model to use only the transcript, to omit empty sections, and to leave out owners/dates that aren't stated.
- **Files:** the upload is saved under a server-chosen name, converted to a temporary 16 kHz mono WAV with FFmpeg, and both copies are deleted when processing ends (success or failure). Your original file is never modified.
- **Safety:** extension and size are checked on the server; user-supplied names are never used as paths; FFmpeg is called without a shell; errors shown to the browser are short messages, never stack traces; CORS is limited to `CORS_ORIGINS`; model output is rendered with `skipHtml`. There's no authentication, so keep it on localhost.

---

## 6. Push to a new GitHub repository

```powershell
cd ai-video-assistant
git init
git add .
git status                 # check that .env, uploads\, .venv and node_modules are NOT listed
git commit -m "Initial commit"
git branch -M main
# create an empty repo on github.com (no README), then:
git remote add origin https://github.com/<your-user>/<new-repo>.git
git push -u origin main
```

`.gitignore` already excludes `.env`, virtual environments, `node_modules`, build output, uploads and vector databases.

## 7. Deploying later (read this first)

**The backend depends on local services**: Whisper needs a lot of CPU/GPU and RAM, and the LLM comes from an Ollama server. A default Render web service has neither a GPU nor Ollama, so *the backend will start but analyses will fail with "Ollama isn't reachable"* unless you point it at a reachable Ollama:

- Set `OLLAMA_BASE_URL` to an Ollama server you run yourself (a GPU VM, or your home PC through a secure tunnel). Don't expose an unauthenticated Ollama port publicly.
- Whisper `small` on a CPU-only host is slow; use `WHISPER_MODEL=base` or `tiny`, and pick a plan with enough RAM (≥ 2 GB, more for `small`). `ffmpeg` must be installed on the host. It may not be present on a plain Python runtime, so check your host's docs (a Docker-based deployment is the usual way to add it).
- Because analyses are stored in memory, run **a single instance** and expect results to disappear on redeploy/restart.

Render-style settings, once the above is solved:
- Python version: 3.12 (`.python-version` is included)
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
- Environment: `CORS_ORIGINS=https://<your-frontend-host>`, `OLLAMA_BASE_URL=...`

**Frontend** (static hosting such as Render Static Site, Netlify or Vercel): root `frontend`, build `npm install && npm run build`, publish `dist`, and set `VITE_API_BASE_URL=https://<your-backend-url>` *before* building (Vite bakes it in).

The earlier `No module named 'pyaudioop'` error came from Pydub on Python 3.13+ (the stdlib `audioop` module was removed). This project doesn't use Pydub at all, and targets Python 3.12.

---

## 8. Verification report

Run in the build environment (Linux, **Python 3.12.3**, Node 22), with dependencies installed from `requirements.txt` into a fresh virtual environment (`pip check`: no broken requirements).

**Passed**
- 37 pytest tests: API validation, uploads, size/extension limits, cleanup, error mapping, session isolation, eviction, CORS; real FFmpeg conversion of audio and video (including no-audio and corrupt files); the real `langchain-ollama` client against a fake Ollama HTTP server (thinking flag sent, `<think>` stripped, chunk + merge, retry when thinking is rejected, unreachable/missing-model errors); real ChromaDB retrieval and per-analysis isolation (with a stand-in embedding function); and one full upload → FFmpeg → pipeline → results → chat run through `api.py`.
- Importing `api` loads none of Whisper, PyTorch, LangChain-Ollama or ChromaDB (no model work at startup).
- `python api.py` and `python -m uvicorn api:app` both start and answer `GET /`; an upload without Ollama returns the clear `503` message.
- Frontend: `tsc` type-check and `vite build` succeed. A scripted Chromium run (19 checks, desktop + mobile width) against the real API covered empty states, file validation, upload, processing state, all result tabs, Markdown tables/bold with raw HTML dropped, copy, chat, history and refresh. No console errors.
- Frontend and backend endpoint paths match; the API base URL is set only in `frontend/src/services/api.ts` (`VITE_API_BASE_URL`).
- No YouTube/URL-ingestion code, no Pydub, no API keys; ZIP contains no `.env`, `node_modules`, virtualenv, databases or model files.

**Not verified here — please test on your machine**
- Real **Whisper** transcription (model download is blocked in the build sandbox; only the clean error path was exercised).
- Real **Chroma embedding model** download/use, and real **Qwen3-8B via Ollama** (output quality, speed and VRAM use on your RTX 4050).
- Running on **Windows** and with a CUDA GPU (commands above follow the official installers but weren't run on Windows).
- Deployment to Render or any host.
