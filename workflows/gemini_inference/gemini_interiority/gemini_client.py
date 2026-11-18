"""Gemini client wrapper for Developer API and Vertex AI backends."""

from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Tuple

from google import genai
from google.genai import types as genai_types
from tqdm.auto import tqdm

CLASSIFY_THINKING_LEVEL = "high"
CLASSIFY_INCLUDE_THOUGHTS = True
CLASSIFY_RESPONSE_MIME_TYPE = "text/x.enum"
TRANSIENT_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}
TRANSIENT_ERROR_NAMES = {
    "RESOURCE_EXHAUSTED",
    "INTERNAL",
    "BAD_GATEWAY",
    "UNAVAILABLE",
    "GATEWAY_TIMEOUT",
    "DEADLINE_EXCEEDED",
    "CONNECTION RESET",
    "CONNECTION ABORTED",
    "TIMED OUT",
}


@dataclass
class GeminiConfig:
    backend: str = "developer"
    api_key: str | None = None
    api_key_env: str = "GEMINI_API_KEY"
    project: str | None = None
    location: str | None = None
    max_output_tokens: int = 4096
    request_timeout_seconds: float | None = 60.0
    retry_sleep_seconds: float = 2.0
    request_delay_seconds: float = 0.0
    max_retries: int = 5
    classify_thinking_level: str = CLASSIFY_THINKING_LEVEL


class GeminiRunnerClient:
    """Small wrapper around the official google-genai SDK."""

    def __init__(self, config: GeminiConfig):
        self.config = config
        self.client = self._make_client(config)
        self.last_usage = self.empty_usage()
        self.last_thoughts = ""
        self.last_batch_thoughts: list[str] = []
        self.last_attempt_count = 0
        self.last_request_failed = False
        self.last_retry_exhausted = False
        self.last_error_message = ""
        self.last_batch_request_metadata: list[dict] = []

    @staticmethod
    def empty_usage() -> dict[str, int]:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "total_tokens": 0,
        }

    @classmethod
    def usage_from_response(cls, response) -> dict[str, int]:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return cls.empty_usage()

        input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        candidate_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        thinking_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)
        total_tokens = int(getattr(usage, "total_token_count", 0) or 0)
        return {
            "input_tokens": input_tokens,
            "output_tokens": candidate_tokens + thinking_tokens,
            "thinking_tokens": thinking_tokens,
            "total_tokens": total_tokens,
        }

    @staticmethod
    def _make_client(config: GeminiConfig):
        backend = config.backend.lower()
        timeout_seconds = config.request_timeout_seconds
        http_options = (
            genai_types.HttpOptions(timeout=int(timeout_seconds * 1000))
            if timeout_seconds is not None and timeout_seconds > 0
            else None
        )
        if backend == "developer":
            api_key = config.api_key or os.getenv(config.api_key_env)
            if not api_key:
                raise ValueError(
                    f"Missing Gemini API key. Set {config.api_key_env} or pass --api-key-env/--api-key."
                )
            return genai.Client(api_key=api_key, http_options=http_options)

        if backend == "vertex":
            project = config.project or os.getenv("GOOGLE_CLOUD_PROJECT")
            location = config.location or os.getenv("GOOGLE_CLOUD_LOCATION")
            if not project or not location:
                raise ValueError(
                    "Vertex backend needs --project and --location, "
                    "or GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION."
                )
            return genai.Client(
                vertexai=True,
                project=project,
                location=location,
                http_options=http_options,
            )

        raise ValueError("backend must be one of: developer, vertex")

    def generate(self, model: str, sys_msg: str, user_msg: str, max_tokens: int | None = None) -> str:
        self.last_usage = self.empty_usage()
        self.last_thoughts = ""
        config_kwargs = {
            "system_instruction": sys_msg,
            "max_output_tokens": max(max_tokens or self.config.max_output_tokens, 100),
        }

        response = self.client.models.generate_content(
            model=model,
            contents=user_msg,
            config=genai_types.GenerateContentConfig(**config_kwargs),
        )
        self.last_usage = self.usage_from_response(response)

        if not response.candidates:
            feedback = getattr(response, "prompt_feedback", None)
            block_reason = getattr(feedback, "block_reason", "unknown") if feedback else "unknown"
            return f"ERROR: Prompt blocked - {block_reason}"

        candidate = response.candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
        text = response.text or ""

        if text.strip():
            return text.strip()

        finish_name = getattr(finish_reason, "name", None)
        if finish_name and finish_name != "STOP":
            return f"ERROR: finish_reason={finish_name}"

        return "ERROR: empty response text"

    @staticmethod
    def _label_enum(labels: Iterable[str]) -> type[Enum]:
        members = {}
        for label in labels:
            value = str(label).strip().upper()
            if not value:
                continue
            name = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "LABEL"
            if name[0].isdigit():
                name = f"LABEL_{name}"
            members[name] = value
        return Enum("GeminiLabelEnum", members, type=str)

    def generate_label(
        self,
        model: str,
        sys_msg: str,
        user_msg: str,
        labels: Iterable[str],
        max_tokens: int | None = None,
    ) -> str:
        self.last_usage = self.empty_usage()
        self.last_thoughts = ""
        config_kwargs = {
            "system_instruction": sys_msg,
            "max_output_tokens": max(max_tokens or self.config.max_output_tokens, 100),
            "response_mime_type": CLASSIFY_RESPONSE_MIME_TYPE,
            "response_schema": self._label_enum(labels),
            "thinking_config": genai_types.ThinkingConfig(
                thinking_level=self.config.classify_thinking_level,
                include_thoughts=CLASSIFY_INCLUDE_THOUGHTS,
            ),
        }

        response = self.client.models.generate_content(
            model=model,
            contents=user_msg,
            config=genai_types.GenerateContentConfig(**config_kwargs),
        )
        self.last_usage = self.usage_from_response(response)

        if not response.candidates:
            feedback = getattr(response, "prompt_feedback", None)
            block_reason = getattr(feedback, "block_reason", "unknown") if feedback else "unknown"
            return f"ERROR: Prompt blocked - {block_reason}"

        candidate = response.candidates[0]
        label_parts = []
        thought_parts = []
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None) or ""
            if not part_text:
                continue
            if bool(getattr(part, "thought", False)):
                thought_parts.append(part_text)
            else:
                label_parts.append(part_text)

        self.last_thoughts = "\n".join(text.strip() for text in thought_parts if text.strip())
        label_text = "".join(label_parts).strip()
        if label_text:
            return label_text

        finish_reason = getattr(candidate, "finish_reason", None)
        finish_name = getattr(finish_reason, "name", None)
        if finish_name and finish_name != "STOP":
            return f"ERROR: finish_reason={finish_name}"

        fallback_text = response.text or ""
        if fallback_text.strip():
            return fallback_text.strip()

        return "ERROR: empty response text"

    @staticmethod
    def _explicit_retry_delay(exc: Exception) -> float | None:
        text = str(exc)
        patterns = [
            r"retryDelay['\"]?\s*:\s*['\"]?([0-9.]+)s",
            r"Please retry in ([0-9.]+)s",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return float(match.group(1))
        return None

    @staticmethod
    def _is_transient_error(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        try:
            if int(status_code) in TRANSIENT_HTTP_STATUS_CODES:
                return True
        except (TypeError, ValueError):
            pass

        text = str(exc).upper()
        exception_name = type(exc).__name__.upper()
        if any(marker in exception_name for marker in ("TIMEOUT", "CONNECTION", "CONNECTERROR", "NETWORK")):
            return True
        if any(name in text for name in TRANSIENT_ERROR_NAMES):
            return True
        return any(re.search(rf"(?<!\d){status}(?!\d)", text) for status in TRANSIENT_HTTP_STATUS_CODES)

    def _retry_sleep_seconds(self, exc: Exception, retry_index: int) -> float:
        exponential_delay = self.config.retry_sleep_seconds * (2 ** retry_index)
        explicit_delay = self._explicit_retry_delay(exc)
        base_delay = max(exponential_delay, explicit_delay or 0.0)
        jitter = random.uniform(0.0, base_delay * 0.25)
        return base_delay + jitter

    def batch_generate(
        self,
        model: str,
        prompts: Iterable[Tuple[str, str]],
        max_tokens: int | None = None,
        show_progress: bool = True,
        retry_on_error: bool = True,
        classify: bool = False,
        response_labels: Iterable[str] | None = None,
    ) -> List[str]:
        results = []
        self.last_batch_thoughts = []
        self.last_batch_request_metadata = []
        iterator = list(prompts)
        progress = tqdm(iterator, desc=f"[{model}]") if show_progress else iterator
        for sys_msg, user_msg in progress:
            result = ""
            self.last_thoughts = ""
            self.last_attempt_count = 0
            self.last_request_failed = False
            self.last_retry_exhausted = False
            self.last_error_message = ""
            for attempt in range(self.config.max_retries + 1):
                self.last_attempt_count = attempt + 1
                try:
                    if classify:
                        result = self.generate_label(
                            model,
                            sys_msg,
                            user_msg,
                            labels=response_labels or ("HIGH", "LOW", "NONE"),
                            max_tokens=max_tokens,
                        )
                    else:
                        result = self.generate(model, sys_msg, user_msg, max_tokens=max_tokens)
                    self.last_request_failed = result.startswith("ERROR:")
                    self.last_error_message = result if self.last_request_failed else ""
                    break
                except Exception as exc:
                    transient = self._is_transient_error(exc)
                    self.last_error_message = str(exc)
                    if not retry_on_error or not transient or attempt >= self.config.max_retries:
                        result = f"ERROR: {exc}"
                        self.last_thoughts = ""
                        self.last_request_failed = True
                        self.last_retry_exhausted = transient and attempt >= self.config.max_retries
                        break
                    sleep_seconds = self._retry_sleep_seconds(exc, attempt)
                    print(
                        f"[{model}] transient API error on attempt {attempt + 1}/"
                        f"{self.config.max_retries + 1}; retrying in {sleep_seconds:.1f}s: {exc}",
                        flush=True,
                    )
                    time.sleep(sleep_seconds)
            if self.config.request_delay_seconds > 0:
                time.sleep(self.config.request_delay_seconds)
            results.append(result)
            self.last_batch_thoughts.append(self.last_thoughts)
            self.last_batch_request_metadata.append(
                {
                    "attempts": self.last_attempt_count,
                    "failed": self.last_request_failed,
                    "retry_exhausted": self.last_retry_exhausted,
                    "error_message": self.last_error_message,
                }
            )
        return results
