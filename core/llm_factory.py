"""Shared, lazily-created Ollama chat model for local or cloud inference."""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.request
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "qwen3:8b"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_CLOUD_BASE_URL = "https://ollama.com"

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(
    r"^\s*```(?:markdown|md)?\s*\n(.*?)\n```\s*$",
    re.DOTALL,
)

_llm_lock = threading.Lock()
_llm = None
_thinking_disabled = True


def is_cloud() -> bool:
    return os.getenv("OLLAMA_CLOUD", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def model_name() -> str:
    default = "gemma4:31b-cloud" if is_cloud() else DEFAULT_MODEL
    return os.getenv("OLLAMA_MODEL", default).strip()


def base_url() -> str:
    default = DEFAULT_CLOUD_BASE_URL if is_cloud() else DEFAULT_BASE_URL
    return os.getenv("OLLAMA_BASE_URL", default).rstrip("/")


def _api_key() -> str:
    return os.getenv("OLLAMA_API_KEY", "").strip()


def _auth_headers() -> dict[str, str]:
    if not is_cloud():
        return {}

    key = _api_key()
    if not key:
        from core import PipelineError

        raise PipelineError(
            "Ollama Cloud is enabled but OLLAMA_API_KEY is missing. "
            "Configure it in your backend environment."
        )

    return {"Authorization": f"Bearer {key}"}


def strip_thinking(text: str) -> str:
    """Remove thinking blocks and wrapping Markdown code fences."""
    if not isinstance(text, str):
        text = str(text or "")

    text = _THINK_BLOCK.sub("", text)
    text = _THINK_OPEN.sub("", text).strip()

    match = _FENCE.match(text)
    return match.group(1).strip() if match else text


def _build_llm(disable_thinking: bool):
    from langchain_ollama import ChatOllama

    kwargs = {
        "model": model_name(),
        "base_url": base_url(),
        "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.2")),
        "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "4096")),
    }

    if is_cloud():
        # langchain-ollama passes these headers to its HTTP client.
        kwargs["client_kwargs"] = {"headers": _auth_headers()}

    if (
        disable_thinking
        and os.getenv("OLLAMA_DISABLE_THINKING", "true").lower() != "false"
    ):
        try:
            return ChatOllama(**kwargs, reasoning=False)
        except (TypeError, ValueError):
            pass

    return ChatOllama(**kwargs)


def get_shared_llm():
    """Return the cached model, creating it on first use."""
    global _llm

    if _llm is None:
        _llm = _build_llm(_thinking_disabled)

    return _llm


def reset_shared_llm() -> None:
    global _llm
    _llm = None


def generate(runnable_factory, inputs) -> str:
    """Run an existing LangChain pipeline and return cleaned text."""
    from core import PipelineError

    global _thinking_disabled

    with _llm_lock:
        for attempt in (1, 2):
            try:
                result = runnable_factory(get_shared_llm()).invoke(inputs)
                return strip_thinking(getattr(result, "content", result))

            except Exception as exc:
                message = str(exc).lower()

                if attempt == 1 and _thinking_disabled and "think" in message:
                    _thinking_disabled = False
                    reset_shared_llm()
                    continue

                if any(
                    term in message
                    for term in (
                        "connect",
                        "connection refused",
                        "timed out",
                        "name resolution",
                        "network is unreachable",
                    )
                ):
                    raise PipelineError(
                        f"Could not reach the Ollama endpoint at {base_url()}. "
                        "Check the endpoint and network connection."
                    ) from exc

                if "401" in message or "unauthorized" in message:
                    raise PipelineError(
                        "Ollama authentication failed. Check OLLAMA_API_KEY."
                    ) from exc

                if "403" in message or "forbidden" in message:
                    raise PipelineError(
                        "Ollama denied access to this model. Check your cloud "
                        "plan, model availability, and usage limits."
                    ) from exc

                if "not found" in message or "pull" in message:
                    if is_cloud():
                        detail = (
                            f"Cloud model '{model_name()}' was not found or "
                            "is unavailable to this account."
                        )
                    else:
                        detail = (
                            f"Local model '{model_name()}' may not be installed. "
                            f"Run: ollama pull {model_name()}"
                        )

                    raise PipelineError(detail) from exc

                raise PipelineError(
                    "The language model failed to respond. Check the model "
                    "configuration and server logs."
                ) from exc

    raise PipelineError("The language model failed to respond.")


def check_ollama() -> Optional[str]:
    """Check local Ollama installation or cloud configuration."""

    if is_cloud():
        if not _api_key():
            return (
                "Ollama Cloud is enabled, but OLLAMA_API_KEY is missing. "
                "Configure it in your backend environment."
            )

        # Cloud model availability is verified by the actual chat request.
        # Do not reject a working cloud model based only on /api/tags.
        return None

    url = f"{base_url()}/api/tags"

    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.load(response)

    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return (
            f"Local Ollama isn't reachable at {base_url()}. Start Ollama and try again."
        )

    names = {model.get("name", "") for model in data.get("models", [])}

    wanted = model_name()

    if wanted not in names and f"{wanted}:latest" not in names:
        return f"The local model '{wanted}' isn't installed. Run: ollama pull {wanted}"

    return None
