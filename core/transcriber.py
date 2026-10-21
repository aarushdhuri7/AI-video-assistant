"""Cloud-based transcription using Sarvam Saaras v4."""

import gc
import os
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from core import PipelineError

load_dotenv()

LANGUAGES = {
    "english": "en-IN",
    "hindi": "hi-IN",
    "auto": "unknown",
    "hinglish": "hi-IN",
}

SARVAM_MODEL = os.getenv("SARVAM_MODEL", "saaras:v2.5")


def _transcribe_with_sarvam(
    audio_path: Path,
    language: str,
) -> str:
    """Transcribe audio through Sarvam's cloud API in small chunks."""

    api_key = os.getenv("SARVAM_API_KEY")

    if not api_key:
        raise PipelineError(
            "SARVAM_API_KEY is missing. Add it to the backend environment."
        )

    if not audio_path.is_file():
        raise PipelineError("The audio file could not be found.")

    try:
        from sarvamai import SarvamAI
    except ImportError as exc:
        raise PipelineError(
            "Sarvam SDK is not installed. Install it with: pip install sarvamai"
        ) from exc

    try:
        client = SarvamAI(api_subscription_key=api_key)
        transcripts = []

        # Keep chunks below Sarvam's 30-second synchronous API limit.
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
                "-c:a",
                "pcm_s16le",
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
                raise PipelineError("FFmpeg could not extract audio from this file.")

            for index, chunk_path in enumerate(chunks, start=1):
                try:
                    with chunk_path.open("rb") as audio_file:
                        response = client.speech_to_text.translate(
                            file=audio_file,
                            model="saaras:v2.5",
                        )

                    transcript = (getattr(response, "transcript", "") or "").strip()

                    if transcript:
                        transcripts.append(transcript)

                    del response
                    gc.collect()

                except Exception as exc:
                    raise PipelineError(
                        f"Sarvam transcription failed on chunk "
                        f"{index} of {len(chunks)}. Check your Sarvam "
                        "API access, language settings, and audio format."
                    ) from exc

        text = " ".join(transcripts).strip()

        if not text:
            raise PipelineError("Sarvam did not detect any speech in this file.")

        return text

    except PipelineError:
        raise

    except FileNotFoundError as exc:
        raise PipelineError(
            "FFmpeg was not found. Install FFmpeg and make it available "
            "in the backend environment."
        ) from exc

    except subprocess.CalledProcessError as exc:
        raise PipelineError(
            "FFmpeg could not extract audio from the uploaded file."
        ) from exc

    except Exception as exc:
        raise PipelineError(
            "Sarvam transcription failed. Check the API configuration and backend logs."
        ) from exc


def transcribe_audio(
    audio_path: Path,
    language: str = "english",
) -> str:
    """
    Transcribe audio using Sarvam cloud transcription.

    Supported choices:
      - english: English transcription
      - hindi: Hindi transcription
      - auto: Automatic language detection
      - hinglish: Code-mixed Hindi-English transcription
    """

    key = (language or "english").strip().lower()

    if key not in LANGUAGES:
        raise PipelineError(
            f"Unsupported language '{language}'. Choose one of: {', '.join(LANGUAGES)}."
        )

    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise PipelineError("The audio file could not be found.")

    return _transcribe_with_sarvam(audio_path, key)
