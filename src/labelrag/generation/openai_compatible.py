"""OpenAI-compatible synchronous answer generator."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, cast
from urllib import error, request

from labelrag.generation.generator import GeneratedAnswer


@dataclass(slots=True)
class OpenAICompatibleConfig:
    """Configuration for OpenAI-compatible chat completion endpoints."""

    model: str
    api_key: str | None = None
    api_key_env_var: str = "OPENAI_API_KEY"
    base_url: str = "https://api.openai.com/v1"
    organization: str | None = None
    timeout_seconds: float = 30.0
    temperature: float = 0.0
    max_tokens: int | None = None
    system_prompt: str = "Answer the question using the provided context."


class OpenAICompatibleAnswerGenerator:
    """Synchronous answer generator for OpenAI-compatible chat APIs."""

    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self.config = config

    def generate(self, question: str, context: str) -> GeneratedAnswer:
        """Generate an answer via a OpenAI-compatible chat completion endpoint."""

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self.config.system_prompt},
                {
                    "role": "user",
                    "content": f"Question:\n{question}\n\nContext:\n{context}",
                },
            ],
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens

        response = self._post_json(
            url=_chat_completions_url(self.config.base_url),
            payload=payload,
        )
        return GeneratedAnswer(
            text=_extract_message_text(response),
            metadata={
                "model": _extract_string(response.get("model")) or self.config.model,
                "usage": _extract_usage(response.get("usage")),
            },
        )

    def _post_json(self, *, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON POST request and return the decoded JSON object."""

        api_key = self._resolve_api_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.config.organization:
            headers["OpenAI-Organization"] = self.config.organization

        http_request = request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=self.config.timeout_seconds) as response:
                return _load_json_response(response.read())
        except error.HTTPError as exc:
            raise RuntimeError(_format_http_error(exc)) from exc
        except error.URLError as exc:
            raise RuntimeError(f"Provider request failed: {exc.reason}") from exc

    def _resolve_api_key(self) -> str:
        """Resolve the API key from config or environment."""

        if self.config.api_key:
            return self.config.api_key

        api_key = os.getenv(self.config.api_key_env_var)
        if api_key:
            return api_key
        raise RuntimeError(
            f"Missing API key. Set `{self.config.api_key_env_var}` or pass `api_key` explicitly."
        )


def _load_json_response(data: bytes) -> dict[str, Any]:
    """Decode a JSON object response."""

    decoded = json.loads(data.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError("Provider response must be a JSON object.")
    return cast(dict[str, Any], decoded)


def _extract_message_text(response: dict[str, Any]) -> str:
    """Extract assistant message text from a chat completion response."""

    choices_value = response.get("choices")
    if not isinstance(choices_value, list) or not choices_value:
        raise RuntimeError("Provider response does not contain any choices.")
    choices = cast(list[object], choices_value)

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError("Provider response contains an invalid choice payload.")
    first_choice_dict = cast(dict[str, Any], first_choice)

    message = first_choice_dict.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("Provider response does not contain a message object.")
    message_dict = cast(dict[str, Any], message)

    content = message_dict.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _join_content_parts(cast(list[object], content))
    raise RuntimeError("Provider response content must be a string or a content-part list.")


def _join_content_parts(parts: list[object]) -> str:
    """Join text content parts from compatible structured responses."""

    texts: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        part_dict = cast(dict[str, Any], part)
        if part_dict.get("type") != "text":
            continue
        text = part_dict.get("text")
        if isinstance(text, str):
            texts.append(text)
    return "".join(texts)


def _extract_usage(value: object) -> dict[str, int]:
    """Extract integer usage metadata when present."""

    if not isinstance(value, dict):
        return {}

    usage: dict[str, int] = {}
    for key, item in cast(dict[object, object], value).items():
        if isinstance(key, str) and isinstance(item, int):
            usage[key] = item
    return usage


def _extract_string(value: object) -> str | None:
    """Extract a string value when present."""

    if isinstance(value, str):
        return value
    return None


def _format_http_error(exc: error.HTTPError) -> str:
    """Format a useful provider HTTP error message."""

    try:
        body = exc.read().decode("utf-8")
    except Exception:
        body = ""
    detail = f": {body}" if body else ""
    return f"Provider request failed with HTTP {exc.code}{detail}"


def _chat_completions_url(base_url: str) -> str:
    """Normalize a base URL or full chat-completions endpoint into a request URL."""

    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"
