"""Shared config: LLM providers, client helpers and Phoenix tracing setup.

Two providers are supported and can be switched with LLM_PROVIDER (or `--provider` on the CLIs):

  openrouter  OpenAI gpt-6.2-sol via OpenRouter: native tool calling + strict JSON-schema outputs
  librechat   the LibreChat agent endpoint: no tool/structured-output support, so the agent falls
              back to a JSON action protocol parsed from plain text
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv
from openai import APIStatusError, AsyncOpenAI

load_dotenv()

PHOENIX_BASE_URL = os.environ.get("PHOENIX_BASE_URL", "http://localhost:6006")
PROD_PROJECT = "shopnova-support-prod"

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    api_key: str
    agent_model: str
    judge_model: str
    native_tools: bool          # supports OpenAI `tools` / `tool_calls`
    structured_output: bool     # supports response_format=json_schema
    headers: tuple = ()


def _providers() -> dict[str, Provider]:
    out = {}
    if os.environ.get("OPENROUTER_API_KEY"):
        model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-6.2-sol")
        out["openrouter"] = Provider(
            name="openrouter",
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.environ["OPENROUTER_API_KEY"],
            agent_model=model,
            judge_model=os.environ.get("OPENROUTER_JUDGE_MODEL", model),
            native_tools=True,
            structured_output=True,
            headers=(("X-Title", "ShopNova Eval Demo"),),
        )
    if os.environ.get("LIBRECHAT_API_KEY"):
        model = os.environ.get("LIBRECHAT_MODEL", "agent__d2PA6a4YSYjSN1SzzbRl")
        out["librechat"] = Provider(
            name="librechat",
            base_url=os.environ["LIBRECHAT_BASE_URL"],
            api_key=os.environ["LIBRECHAT_API_KEY"],
            agent_model=model,
            judge_model=os.environ.get("LIBRECHAT_JUDGE_MODEL", model),
            native_tools=False,
            structured_output=False,
        )
    return out


PROVIDERS = _providers()
_active = os.environ.get("LLM_PROVIDER", "openrouter" if "openrouter" in PROVIDERS else "librechat")


def set_provider(name: str) -> None:
    global _active
    if name not in PROVIDERS:
        raise ValueError(f"provider '{name}' not configured in .env (have: {list(PROVIDERS)})")
    _active = name


def provider() -> Provider:
    return PROVIDERS[_active]


@lru_cache
def _client(name: str) -> AsyncOpenAI:
    p = PROVIDERS[name]
    return AsyncOpenAI(base_url=p.base_url, api_key=p.api_key, timeout=120, max_retries=3,
                       default_headers=dict(p.headers) or None)


def llm_client() -> AsyncOpenAI:
    return _client(_active)


class LLMResponseError(Exception):
    pass


DEFAULT_MAX_TOKENS = 1500  # also caps OpenRouter's per-request credit reservation (402s under concurrency)


async def chat_completion(messages: list[dict], model: str, **kwargs):
    """One chat.completions call with response validation + retry. Returns the assistant message."""
    if provider().name == "openrouter":
        kwargs.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
    last: Exception | None = None
    for attempt in range(6):
        try:
            try:
                resp = await llm_client().chat.completions.create(model=model, messages=messages, stream=False, **kwargs)
            except APIStatusError as e:
                if e.status_code in (402, 429, 502, 503):  # credit reservation / rate limit / upstream hiccup
                    raise LLMResponseError(f"HTTP {e.status_code}: {str(e)[:160]}") from e
                raise
            if isinstance(resp, str) or not getattr(resp, "choices", None):
                raise LLMResponseError(f"unexpected response body: {str(resp)[:300]!r}")
            msg = resp.choices[0].message
            if (msg.content or "").lstrip().startswith("event: "):
                # the LibreChat gateway occasionally leaks its internal SSE stream into the content
                raise LLMResponseError(f"gateway leaked SSE stream: {msg.content[:120]!r}")
            return msg
        except LLMResponseError as e:
            last = e
            log.warning("LLM call attempt %d failed: %s", attempt + 1, e)
            await asyncio.sleep(2.0 * (attempt + 1))
    raise last  # type: ignore[misc]


async def chat(messages: list[dict], model: str, *, json_schema: dict | None = None, schema_name: str = "output") -> str:
    """Text completion. With `json_schema`, uses strict structured output when the provider supports it."""
    kwargs = {}
    if json_schema is not None and provider().structured_output:
        kwargs["response_format"] = {"type": "json_schema",
                                     "json_schema": {"name": schema_name, "strict": True, "schema": json_schema}}
    msg = await chat_completion(messages, model, **kwargs)
    return msg.content or ""


_tracer_provider = None


def setup_tracing(project_name: str = PROD_PROJECT):
    """Register Phoenix as the OTel backend and auto-instrument the OpenAI SDK.

    Every chat.completions call becomes an LLM span (messages, tool schemas, tool calls, tokens);
    the pipeline adds AGENT / CHAIN / TOOL / RETRIEVER / GUARDRAIL spans around them.
    """
    global _tracer_provider
    if _tracer_provider is None:
        from phoenix.otel import register

        _tracer_provider = register(
            project_name=project_name,
            endpoint=f"{PHOENIX_BASE_URL}/v1/traces",
            auto_instrument=True,  # picks up openinference-instrumentation-openai
            batch=True,
            verbose=False,
        )
    return _tracer_provider


def get_tracer():
    return setup_tracing().get_tracer("shopnova.support_agent")


def flush_traces() -> None:
    if _tracer_provider is not None:
        _tracer_provider.force_flush()
