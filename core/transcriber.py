"""Whisper and Sarvam transcription with lazy model loading."""

from __future__ import annotations

import gc
import os
import threading
import tempfile
import subprocess
from pathlib import Path

from dotenv import load_dotenv

from core import PipelineError

load_dotenv()

# UI/CLI language name -> Whisper language code.
LANGUAGES = {
    "english": "en",
    "hindi": "hi",
    "auto": None,
    "hinglish": "hinglish",
}

_model = None
_model_lock = threading.Lock()

SARVAM_MODEL = os.getenv("SARVAM_MODEL", "saaras:v4")


def whisper_model_name() -> str:
    # tiny is significantly lighter than small on limited-memory hosts.
    return os.getenv("WHISPER_MODEL", "tiny")


def get_model():
    """Load the Whisper model once, when first needed."""
    global _model

    with _model_lock:
        if _model is None:
            try:
                import whisper
            except ImportError as exc:
                raise PipelineError(
                    "Whisper isn't installed. Run: pip install -r requirements.txt"
                ) from exc

            try:
                _model = whisper.load_model(
                    whisper_model_name(),
                    device="cpu",
                )
            except Exception as exc:
                raise PipelineError(
                    f"Could not load the Whisper model "
                    f"'{whisper_model_name()}'. Check available memory."
                ) from exc

    return _model


def _transcribe_with_whisper(audio_path: Path, language: str) -> str:
    """Transcribe using local Whisper with reduced memory usage."""
    import torch

    model = get_model()

    try:
        with torch.inference_mode():
            result = model.transcribe(
                str(audio_path),
                language=LANGUAGES[language],
                task="transcribe",
                fp16=False,
                verbose=False,
            )

        text = " ".join((result.get("text") or "").split())
        del result
        gc.collect()

    except Exception as exc:
        gc.collect()
        raise PipelineError(
            "Whisper transcription failed. The audio may be corrupt, "
            "unreadable, or too large for the available memory."
        ) from exc

    if not text:
        raise PipelineError("No speech was detected in this file.")

    return text


def _transcribe_with_sarvam(audio_path: Path) -> str:
    """Transcribe Hindi-English mixed speech using Sarvam."""
    api_key = os.getenv("SARVAM_API_KEY")

    if not api_key:
        raise PipelineError(
            "Hinglish transcription requires SARVAM_API_KEY. "
            "Add your Sarvam API key to the backend environment."
        )

    if not audio_path.is_file():
        raise PipelineError("The audio file could not be found.")

    try:
        from sarvamai import SarvamAI
    except ImportError as exc:
        raise PipelineError(
            "Sarvam SDK isn't installed. Run: pip install sarvamai"
        ) from exc

    try:
        client = SarvamAI(api_subscription_key=api_key)
        transcripts = []
        chunk_seconds = 25

        with tempfile.TemporaryDirectory(prefix="sarvam_transcription_") as temp_dir:
            output_pattern = str(Path(temp_dir) / "chunk_%04d.wav")

            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(audio_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "segment",
                "-segment_time",
                str(chunk_seconds),
                "-reset_timestamps",
                "1",
                output_pattern,
            ]

            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )

            chunks = sorted(Path(temp_dir).glob("chunk_*.wav"))

            if not chunks:
                raise PipelineError("FFmpeg could not prepare audio for Sarvam.")

            for index, chunk_path in enumerate(chunks, start=1):
                try:
                    with chunk_path.open("rb") as audio_file:
                        response = client.speech_to_text.transcribe(
                            file=audio_file,
                            model=SARVAM_MODEL,
                            mode="codemix",
                            language_code="hi-IN",
                        )

                    transcript = (getattr(response, "transcript", "") or "").strip()

                    if transcript:
                        transcripts.append(transcript)

                    del response
                    gc.collect()

                except Exception as exc:
                    raise PipelineError(
                        f"Sarvam transcription failed on audio chunk "
                        f"{index} of {len(chunks)}. Check your API key, "
                        "account access, internet connection, and audio format."
                    ) from exc

        text = " ".join(transcripts).strip()

        if not text:
            raise PipelineError("Sarvam did not detect any speech in this file.")

        return text

    except PipelineError:
        raise
    except FileNotFoundError as exc:
        raise PipelineError(
            "FFmpeg was not found. Install FFmpeg and add it to PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise PipelineError("FFmpeg could not prepare the audio for Sarvam.") from exc
    except Exception as exc:
        raise PipelineError(
            "Sarvam transcription failed. Check the API configuration and try again."
        ) from exc


def transcribe_audio(
    audio_path: Path,
    language: str = "english",
) -> str:
    """Transcribe audio using local Whisper or Sarvam."""
    key = (language or "english").strip().lower()

    if key not in LANGUAGES:
        raise PipelineError(
            f"Unsupported language '{language}'. Choose one of: {', '.join(LANGUAGES)}."
        )

    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise PipelineError("The audio file could not be found.")

    if key == "hinglish":
        return _transcribe_with_sarvam(audio_path)

    return _transcribe_with_whisper(audio_path, key)
