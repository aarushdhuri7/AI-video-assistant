"""Action items, key decisions and open questions.

Each extractor runs on the transcript in chunks (so long recordings fit the model's
context), then merges the partial lists with one extra call.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from core.chunking import split_transcript
from core.llm_factory import generate

NONE_WORD = "NONE"

COMMON = (
    "Use ONLY information stated in the transcript. Never invent items, people, dates or details. "
    f"If there is nothing to report, reply with exactly: {NONE_WORD}\n"
    "Output Markdown only: no introduction and no reasoning."
)

ACTIONS = (
    "Extract the action items (tasks someone agreed or was asked to do) from this transcript.\n"
    "Format each as a bullet: `- **Task** (Owner: <name>, Due: <date>)`. "
    "Include Owner and Due ONLY when explicitly stated; otherwise leave them out entirely.\n" + COMMON
)
DECISIONS = (
    "Extract the key decisions that were made or agreed in this transcript, as a bullet list. "
    "Add a short reason only if it is stated.\n" + COMMON
)
QUESTIONS = (
    "Extract the open questions: questions that were raised but not answered, and topics explicitly "
    "left for follow-up. Use a bullet list.\n" + COMMON
)
MERGE = (
    "These are partial lists extracted from consecutive parts of one transcript. Merge them into a single "
    "Markdown bullet list, removing duplicates and keeping the original wording and format. "
    "Do not add anything new.\n" + COMMON
)


def _is_none(text: str) -> bool:
    return not text.strip() or text.strip().strip(".*`").upper() == NONE_WORD


def _run(system: str, text: str) -> str:
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{text}")])
    return generate(lambda llm: prompt | llm, {"text": text})


def _extract(transcript: str, system: str, empty_message: str) -> str:
    if not transcript or not transcript.strip():
        return empty_message
    parts = [r for r in (_run(system, c) for c in split_transcript(transcript)) if not _is_none(r)]
    if not parts:
        return empty_message
    if len(parts) == 1:
        return parts[0]
    merged = _run(MERGE, "\n\n".join(parts))
    return empty_message if _is_none(merged) else merged


def extract_action_items(transcript: str) -> str:
    return _extract(transcript, ACTIONS, "No action items were found in this transcript.")


def extract_key_decisions(transcript: str) -> str:
    return _extract(transcript, DECISIONS, "No key decisions were found in this transcript.")


def extract_questions(transcript: str) -> str:
    return _extract(transcript, QUESTIONS, "No open questions were found in this transcript.")
