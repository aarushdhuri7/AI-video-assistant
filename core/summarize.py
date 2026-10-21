"""Markdown summary and title generation."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from core import PipelineError
from core.chunking import split_transcript
from core.llm_factory import generate


REPORT_SYSTEM = """You create accurate, useful Markdown reports from transcripts
of meetings, lectures, interviews, and other recordings.

STRICT RULES:
- Use ONLY information supported by the transcript.
- Never invent names, facts, numbers, deadlines, decisions, tasks, or answers.
- Distinguish explicit decisions from suggestions and opinions.
- An action item must be a task supported by the transcript. Do not invent an
  owner or deadline when none is stated.
- An open question must be a question that remains unresolved in the transcript.
  Do not invent questions merely to fill the section.
- If the transcript does not provide enough information for a section, write
  "None explicitly identified in the transcript." under that heading.
- Do not confuse missing information with proof that something does not exist.
- Output Markdown only. No introduction, closing remarks, or reasoning.

ALWAYS include all six headings below, in exactly this order.
Never omit a heading, even when its content is unavailable.

## Meeting Summary
Give a concise overview of the recording.

## Main Discussion
List the important topics, explanations, and arguments.

## Key Decisions
List decisions actually made. Distinguish tentative proposals from final decisions.
If none are explicit, say so.

## Action Items
List supported tasks. Include the owner and deadline only when stated.
If no tasks are identified, say so.

## Open Questions
List questions that remain unanswered or issues explicitly left unresolved.
If none are identified, say so.

## Core Takeaway
Give a short paragraph capturing the main conclusion.

Use concise paragraphs and informative bullet points. Preserve important
names, figures, technical terms, and context."""


NOTES_SYSTEM = """Extract concise factual notes from this portion of a transcript.

Preserve:
- Main topics and important explanations
- Names, numbers, dates, and technical terms
- Decisions, distinguishing final decisions from suggestions
- Tasks, stated owners, and stated deadlines
- Questions that remain unanswered and unresolved issues

Use only information supported by this portion of the transcript.
Do not invent missing information or draw unsupported conclusions.
Use compact bullet points. No introduction or closing remarks."""


MERGE_SYSTEM = """Combine notes from consecutive portions of one recording into
one accurate, useful Markdown report.

Follow the requested six-section report structure exactly.
Always include every heading, in this order:
Meeting Summary, Main Discussion, Key Decisions, Action Items,
Open Questions, Core Takeaway.

Use these exact Markdown headings:
## Meeting Summary
## Main Discussion
## Key Decisions
## Action Items
## Open Questions
## Core Takeaway

Rules:
- Use only information supported by the supplied notes.
- Merge duplicate points without losing important details.
- Preserve names, figures, decisions, tasks, owners, and deadlines.
- Do not treat a suggestion as a confirmed decision.
- Do not invent tasks, owners, deadlines, or unresolved questions.
- Under an empty section, write:
  "None explicitly identified in the transcript."
- Meeting Summary and Core Takeaway should be concise paragraphs.
- Use bullet points for Main Discussion, Key Decisions, Action Items,
  and Open Questions.
- Return Markdown only. Do not add an introduction or conclusion."""


def _run(system: str, text: str) -> str:
    """Run one prompt against the configured shared language model."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{text}"),
        ]
    )
    return generate(lambda llm: prompt | llm, {"text": text})


def summarize(transcript: str) -> str:
    """Create a report with all required sections."""
    if not transcript or not transcript.strip():
        raise PipelineError("There is no transcript to summarize.")

    chunks = split_transcript(transcript)

    try:
        if len(chunks) == 1:
            report = _run(REPORT_SYSTEM, chunks[0])
        else:
            # Summarize chunks separately to preserve information from long
            # recordings without exceeding the model's context window.
            notes = [_run(NOTES_SYSTEM, chunk) for chunk in chunks]

            # Consolidate the notes into the final six-section report.
            report = _run(
                MERGE_SYSTEM,
                "\n\n---\n\n".join(notes),
            )

    except PipelineError:
        raise

    except Exception as exc:
        raise PipelineError(
            "Could not generate the report. Check the language model logs."
        ) from exc

    if not report or not report.strip():
        raise PipelineError(
            "The language model returned an empty report. Please try again."
        )

    return report.strip()


def generate_title(transcript: str) -> str:
    """Generate a short title from the beginning of the transcript."""
    if not transcript or not transcript.strip():
        return "Untitled analysis"

    text = _run(
        "Write a descriptive title of no more than 8 words for this recording. "
        "Use only information supported by the transcript. "
        "Return the title only, without quotes or ending punctuation.",
        transcript[:2500],
    )

    title = text.splitlines()[0].strip().strip("\"'*#` ").rstrip(".") if text else ""

    return title or "Untitled analysis"
