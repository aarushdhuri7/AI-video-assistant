"""Whisper and Sarvam transcription with lazy model loading."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from dotenv import load_dotenv

from core import PipelineError

load_dotenv()

# UI/CLI language name -> Whisper language code.
# None means automatic language detection.
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
    return os.getenv("WHISPER_MODEL", "small")


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

            device = os.getenv("WHISPER_DEVICE") or None

            try:
                _model = whisper.load_model(
                    whisper_model_name(),
                    device=device,
                )
            except Exception as exc:
                raise PipelineError(
                    f"Could not load the Whisper model "
                    f"'{whisper_model_name()}'. The first run downloads "
                    "it, so check your internet connection."
                ) from exc

    return _model


def _transcribe_with_whisper(
    audio_path: Path,
    language: str,
) -> str:
    """Transcribe using local Whisper."""
    model = get_model()

    try:
        result = model.transcribe(
            str(audio_path),
            language=LANGUAGES[language],
            task="transcribe",
            fp16=getattr(model.device, "type", "cpu") == "cuda",
        )
    except Exception as exc:
        raise PipelineError(
            "Whisper transcription failed. The audio may be corrupt or unreadable."
        ) from exc

    text = " ".join((result.get("text") or "").split())

    if not text:
        raise PipelineError("No speech was detected in this file.")

    return text


def _transcribe_with_sarvam(audio_path: Path) -> str:
    """Transcribe Hindi-English mixed speech using Sarvam."""
    api_key = os.getenv("SARVAM_API_KEY")

    if not api_key:
        raise PipelineError(
            "Hinglish transcription requires SARVAM_API_KEY. "
            "Add your Sarvam API key to the backend .env file."
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
        import subprocess
        import tempfile

        client = SarvamAI(api_subscription_key=api_key)
        transcripts = []

        # Keep each synchronous request below the short-audio limit.
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

                except Exception as exc:
                    raise PipelineError(
                        f"Sarvam transcription failed on audio "
                        f"chunk {index} of {len(chunks)}. "
                        "Check your API key, account access, "
                        "internet connection and audio format."
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
    """
    Transcribe audio with local Whisper or Sarvam.

    Supported language choices:
      - english: local Whisper, English
      - hindi: local Whisper, Hindi
      - auto: local Whisper, automatic detection
      - hinglish: Sarvam code-mixed transcription
    """
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
