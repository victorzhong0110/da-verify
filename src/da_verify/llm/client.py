"""OpenAI-compatible LLM client with a record-replay cache.

WHY record-replay (earned reason):
  The controlled study reruns the same prompts many times (debugging, re-scoring,
  reproducing a number for the report). Hitting a paid API every time is slow,
  costs money, and — worse — is NON-deterministic, so results wouldn't reproduce.
  We hash each request (model + messages + tools + sampling params) and cache the
  response on disk. Same request -> same answer, for free. This is what lets a
  reviewer reproduce the headline number from a fresh clone without a key.

The cache key intentionally includes temperature: a different sampling setting
is a different experiment and must not collide.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

_ROOT = Path(__file__).resolve().parents[3]


def load_env(env_path: Path = _ROOT / ".env") -> None:
    """Tiny .env loader (no extra dependency). Does not overwrite real env vars."""
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


@dataclass(frozen=True)
class LLMResponse:
    """Normalised, cacheable view of one completion."""

    content: str | None
    tool_calls: list[dict]  # [{"id","name","arguments"(parsed dict),"arguments_raw"(str)}]
    finish_reason: str
    assistant_message: dict  # ready to append back into `messages` for the next turn
    cached: bool = False
    usage: dict = field(default_factory=dict)


class LLMClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        cache_dir: Path = _ROOT / "cache" / "llm",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 120.0,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    @classmethod
    def from_env(cls, **kw) -> "LLMClient":
        load_env()
        provider = os.environ.get("DA_VERIFY_PROVIDER", "minimax").lower()
        key = os.environ[f"{provider.upper()}_API_KEY"]
        base = os.environ[f"{provider.upper()}_BASE_URL"]
        model = os.environ.get("DA_VERIFY_MODEL", "MiniMax-M2.7")
        return cls(model=model, api_key=key, base_url=base, **kw)

    def _cache_key(self, messages: list[dict], tools: list[dict] | None, sample_id: int) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        # sample_id distinguishes the k repeats for pass@k (each must be a fresh
        # draw, not a cache collapse). Omitted at 0 so the W2 temp=0 cache still hits.
        if sample_id:
            payload["sample_id"] = sample_id
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def chat(self, messages: list[dict], tools: list[dict] | None = None, sample_id: int = 0) -> LLMResponse:
        key = self._cache_key(messages, tools, sample_id)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return LLMResponse(**{**data, "cached": True})

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        norm = self._normalise(resp)
        cache_file.write_text(
            json.dumps(
                {
                    "content": norm.content,
                    "tool_calls": norm.tool_calls,
                    "finish_reason": norm.finish_reason,
                    "assistant_message": norm.assistant_message,
                    "usage": norm.usage,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return norm

    @staticmethod
    def _normalise(resp) -> LLMResponse:
        choice = resp.choices[0]
        msg = choice.message
        tool_calls: list[dict] = []
        assistant_tcs: list[dict] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            raw_args = tc.function.arguments or "{}"
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed = None  # malformed tool args — surfaced, not hidden
            tool_calls.append(
                {"id": tc.id, "name": tc.function.name, "arguments": parsed, "arguments_raw": raw_args}
            )
            assistant_tcs.append(
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": raw_args}}
            )
        assistant_message: dict = {"role": "assistant", "content": msg.content or ""}
        if assistant_tcs:
            assistant_message["tool_calls"] = assistant_tcs
        usage = {}
        if getattr(resp, "usage", None):
            usage = {"prompt": resp.usage.prompt_tokens, "completion": resp.usage.completion_tokens}
        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            assistant_message=assistant_message,
            usage=usage,
        )
