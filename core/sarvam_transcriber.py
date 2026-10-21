import os
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "saaras:v2.5")

CHUNK_SECONDS = 25


def _split_audio(audio_path: str, output_dir: str) -> list[str]:
    """Split audio into short WAV chunks using FFmpeg."""
    output_pattern = str(Path(output_dir) / "chunk_%04d.wav")

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
        str(CHUNK_SECONDS),
        "-reset_timestamps",
        "1",
        output_pattern,
    ]

    subprocess.run(command, check=True)

    chunks = sorted(Path(output_dir).glob("chunk_*.wav"))

    if not chunks:
        raise RuntimeError("FFmpeg did not produce any audio chunks.")

    return [str(chunk) for chunk in chunks]


def transcribe_hinglish(audio_path: str) -> str:
    """
    Transcribe Hindi-English mixed speech using Sarvam.
    Returns the combined transcript in code-mixed output.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is missing. Add it to your .env file.")

    if not Path(audio_path).is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

    with tempfile.TemporaryDirectory(prefix="sarvam_") as temp_dir:
        chunks = _split_audio(audio_path, temp_dir)
        transcripts = []

        for index, chunk_path in enumerate(chunks, start=1):
            print(f"Sarvam transcription: chunk {index}/{len(chunks)}")

            with open(chunk_path, "rb") as audio_file:
                response = client.speech_to_text.transcribe(
                    file=audio_file,
                    model=SARVAM_MODEL,
                    mode="codemix",
                    language_code="hi-IN",
                )

            transcript = (response.transcript or "").strip()

            if transcript:
                transcripts.append(transcript)

        if not transcripts:
            raise RuntimeError(
                "Sarvam returned an empty transcript for all audio chunks."
            )

        return "\n".join(transcripts)
