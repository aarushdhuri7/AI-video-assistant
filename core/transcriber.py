"""Cloud-based transcription using Sarvam Saaras v2.5."""

import gc
import os
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from core import PipelineError

load_dotenv()

# Saaras v2.5 automatically detects the spoken language
# and translates speech into English.
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "saaras:v2.5")


def _transcribe_with_sarvam(audio_path: Path, language: str) -> str:
    """Extract audio in short chunks and translate it using Sarvam."""

    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise PipelineError("SARVAM_API_KEY is missing from the backend environment.")

    if not audio_path.is_file():
        raise PipelineError("The audio file could not be found.")

    try:
        from sarvamai import SarvamAI
    except ImportError as exc:
        raise PipelineError(
            "Sarvam SDK is not installed. Add sarvamai to requirements.txt."
        ) from exc

    try:
        client = SarvamAI(api_subscription_key=api_key)
        transcripts = []

        # Sarvam's synchronous endpoint accepts audio up to 30 seconds.
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
                "-c:a",
                "pcm_s16le",
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
                timeout=600,
            )

            chunks = sorted(Path(temp_dir).glob("chunk_*.wav"))

            if not chunks:
                raise PipelineError(
                    "FFmpeg could not extract audio from the uploaded file."
                )

            for index, chunk_path in enumerate(chunks, start=1):
                try:
                    with chunk_path.open("rb") as audio_file:
                        response = client.speech_to_text.translate(
                            file=audio_file,
                            model=SARVAM_MODEL,
                        )

                    transcript = (getattr(response, "transcript", "") or "").strip()

                    if transcript:
                        transcripts.append(transcript)

                    del response
                    gc.collect()

                except Exception as exc:
                    # Include the API error, but never expose the API key.
                    raise PipelineError(
                        f"Sarvam failed on audio chunk {index} "
                        f"of {len(chunks)}. API error: {exc}"
                    ) from exc

        text = " ".join(transcripts).strip()

        if not text:
            raise PipelineError("Sarvam returned no transcript for this audio.")

        return text

    except PipelineError:
        raise

    except FileNotFoundError as exc:
        raise PipelineError(
            "FFmpeg was not found. Install FFmpeg in the backend environment."
        ) from exc

    except subprocess.TimeoutExpired as exc:
        raise PipelineError(
            "Audio extraction took too long. Try a shorter video."
        ) from exc

    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or "").strip()[-1000:]
        raise PipelineError(f"FFmpeg could not extract audio. {details}") from exc

    except Exception as exc:
        raise PipelineError(f"Sarvam transcription failed: {exc}") from exc


def transcribe_audio(
    audio_path: Path,
    language: str = "english",
) -> str:
    """
    Transcribe audio using Sarvam Saaras v2.5.

    Saaras v2.5 automatically detects the input language and
    returns English translation. The language argument is retained
    for compatibility with the existing application.
    """

    supported_languages = {"english", "hindi", "auto", "hinglish"}
    key = (language or "english").strip().lower()

    if key not in supported_languages:
        raise PipelineError(
            f"Unsupported language '{language}'. Choose one of: "
            f"{', '.join(sorted(supported_languages))}."
        )

    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise PipelineError("The audio file could not be found.")

    return _transcribe_with_sarvam(audio_path, key)
