"""Pipeline orchestration, plus a small command-line interface.

    python main.py path/to/recording.mp4
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

from core import PipelineError
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import ask_question, build_rag_chain
from core.summarize import generate_title, summarize
from core.transcriber import transcribe_audio
from utils.audio_processor import extract_audio

load_dotenv()

Progress = Optional[Callable[[str], None]]


def run_pipeline(source, language: str = "english", progress: Progress = None) -> dict:
    """Analyse one audio/video file and return every result in a dictionary.

    ``source`` is a path to a local file. ``progress`` (optional) is called with the
    name of each stage as it starts.
    """

    def step(name: str) -> None:
        if progress:
            progress(name)

    work_dir = Path(tempfile.mkdtemp(prefix="vassist_"))  # temp audio; the original file is never touched
    try:
        step("Extracting audio")
        wav = extract_audio(Path(source), work_dir)

        step("Transcribing")
        transcript = transcribe_audio(wav, language)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    step("Summarizing")
    title = generate_title(transcript)
    summary = summarize(transcript)

    step("Extracting action items, decisions and questions")
    action_items = extract_action_items(transcript)
    key_decisions = extract_key_decisions(transcript)
    open_questions = extract_questions(transcript)

    step("Building chat index")
    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": key_decisions,
        "open_questions": open_questions,
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python main.py <audio-or-video-file> [english|hindi|auto]")
    lang = sys.argv[2] if len(sys.argv) > 2 else "english"
    try:
        result = run_pipeline(sys.argv[1], lang, progress=lambda s: print(f"... {s}"))
    except PipelineError as err:
        sys.exit(f"Error: {err}")
    print(f"\n# {result['title']}\n\n{result['summary']}")
    print(f"\n## Action items\n{result['action_items']}\n\n## Key decisions\n{result['key_decisions']}\n\n## Open questions\n{result['open_questions']}")
    print("\nAsk questions about the transcript (blank line to quit).")
    while (q := input("\nYou: ").strip()):
        try:
            print(f"\nAssistant: {ask_question(result['rag_chain'], q)}")
        except PipelineError as err:
            print(f"Error: {err}")
