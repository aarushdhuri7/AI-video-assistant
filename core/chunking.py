import os

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_transcript(text: str, size: int | None = None, overlap: int = 200) -> list[str]:
    """Split long text into pieces that fit comfortably in the model's context window."""
    size = size or int(os.getenv("CHUNK_CHARS", "5000"))
    if len(text) <= size:
        return [text]
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap).split_text(text)
