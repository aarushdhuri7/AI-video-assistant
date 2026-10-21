"""Question answering over one transcript (RAG).

* Vector store: ChromaDB, in memory (nothing is written to disk).
* Embeddings: Chroma's built-in all-MiniLM-L6-v2 (ONNX, ~80 MB, downloaded once on first use,
  runs on CPU). No second language model is needed.
* Every `build_rag_chain` call creates its own collection, so analyses never share data.
"""
from __future__ import annotations

import uuid

from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core import PipelineError
from core.llm_factory import generate

NOT_FOUND = "I could not find this information in the transcript."
TOP_K = 4

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about a transcript using ONLY the excerpts below.\n"
            f'If the answer is not in the excerpts, reply exactly: "{NOT_FOUND}"\n'
            "Be concise and accurate. Do not guess. Use Markdown if it helps.\n\nExcerpts:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

_client = None


def _get_client():
    global _client
    if _client is None:
        import chromadb
        from chromadb.config import Settings

        _client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    return _client


def _get_embedding_function():
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    return DefaultEmbeddingFunction()


class RagChain:
    """Holds one transcript's vector index. Stays on the server; never sent to the browser."""

    def __init__(self, transcript: str):
        chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(transcript)
        self.name = f"t_{uuid.uuid4().hex}"
        try:
            self._collection = _get_client().create_collection(self.name, embedding_function=_get_embedding_function())
            self._collection.add(documents=chunks, ids=[f"c{i}" for i in range(len(chunks))])
        except Exception as exc:  # noqa: BLE001
            self.close()
            raise PipelineError(
                "Could not build the search index. The embedding model is downloaded on first use, "
                "so check your internet connection."
            ) from exc

    def retrieve(self, question: str) -> list[str]:
        res = self._collection.query(query_texts=[question], n_results=TOP_K)
        return (res.get("documents") or [[]])[0]

    def close(self) -> None:
        try:
            _get_client().delete_collection(self.name)
        except Exception:  # noqa: BLE001
            pass


def build_rag_chain(transcript: str) -> RagChain:
    if not transcript or not transcript.strip():
        raise PipelineError("There is no transcript to search.")
    return RagChain(transcript)


def ask_question(rag_chain: RagChain, question: str) -> str:
    question = (question or "").strip()
    if not question:
        raise PipelineError("Please enter a question.")
    try:
        docs = rag_chain.retrieve(question)
    except Exception as exc:  # noqa: BLE001
        raise PipelineError("Searching the transcript failed.") from exc
    if not docs:
        return NOT_FOUND
    context = "\n\n---\n\n".join(docs)
    return generate(lambda llm: PROMPT | llm, {"context": context, "question": question}) or NOT_FOUND
