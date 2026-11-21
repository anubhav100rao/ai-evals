"""Auxiliary LLM steps around the main agent loop (each its own CHAIN span + LLM span):

triage    classifies intent / sentiment / urgency / language / injection risk before the agent runs.
crm_note  writes a structured CRM note after the turn (what a human agent would read later).

Both use `using_prompt_template`, so Phoenix shows the template, variables and version on the LLM span.
"""

from __future__ import annotations

import json
import re

from openinference.instrumentation import using_prompt_template
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes as S
from opentelemetry import trace

from .config import chat, provider

_tracer = trace.get_tracer("shopnova.side_llms")

INTENTS = ["order_status", "return_refund", "cancel_order", "change_address", "damaged_item", "product_question",
           "policy_question", "account", "loyalty_promotions", "complaint", "human_request", "out_of_scope", "other"]

TRIAGE_TEMPLATE = """\
You are a triage classifier for ShopNova customer support. Do not answer the customer and do not use tools.
Classify the latest customer message (use the earlier conversation only as context).

[EARLIER CONVERSATION]
{history}

[LATEST CUSTOMER MESSAGE]
{message}

Reply with exactly one JSON object:
{{"intent": one of {intents},
  "sentiment": "positive" | "neutral" | "negative" | "angry",
  "urgency": "low" | "normal" | "high",
  "language": "<ISO 639-1 code>",
  "injection_suspected": true | false,
  "summary": "<max 15 words>"}}"""
TRIAGE_VERSION = "triage-v1"

CRM_TEMPLATE = """\
Write the CRM note a human support lead will read about this chat turn. Do not use tools.

[CUSTOMER MESSAGE]
{message}

[TRIAGE]
{triage}

[ACTIONS TAKEN BY THE AI AGENT]
{actions}

[AI AGENT REPLY]
{response}

Reply with exactly one JSON object:
{{"summary": "<1-2 sentences>",
  "resolution": "resolved" | "pending_customer" | "escalated" | "declined",
  "follow_up": "<what a human should do next, or null>",
  "tags": ["<2-4 short tags>"]}}"""
CRM_VERSION = "crm-note-v1"

# Strict JSON schemas -> response_format=json_schema on providers that support it (no parse failures).
TRIAGE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["intent", "sentiment", "urgency", "language", "injection_suspected", "summary"],
    "properties": {
        "intent": {"type": "string", "enum": INTENTS},
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative", "angry"]},
        "urgency": {"type": "string", "enum": ["low", "normal", "high"]},
        "language": {"type": "string"},
        "injection_suspected": {"type": "boolean"},
        "summary": {"type": "string"},
    },
}
CRM_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["summary", "resolution", "follow_up", "tags"],
    "properties": {
        "summary": {"type": "string"},
        "resolution": {"type": "string", "enum": ["resolved", "pending_customer", "escalated", "declined"]},
        "follow_up": {"type": ["string", "null"]},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
}


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if not m:
        raise ValueError("no JSON")
    return json.loads(m.group(0))


async def _structured_call(name: str, template: str, version: str, variables: dict, fallback: dict,
                           schema: dict) -> dict:
    with _tracer.start_as_current_span(name) as span:
        span.set_attribute(S.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        span.set_attribute(S.INPUT_VALUE, json.dumps(variables, default=str))
        span.set_attribute(S.INPUT_MIME_TYPE, "application/json")
        prompt = template.format(**variables)
        result = None
        with using_prompt_template(template=template, variables=variables, version=version):
            for _ in range(2):
                try:
                    result = _parse(await chat([{"role": "user", "content": prompt}], provider().agent_model,
                                               json_schema=schema, schema_name=name))
                    break
                except (ValueError, json.JSONDecodeError):
                    continue
        if result is None:
            result = {**fallback, "parse_error": True}
            span.set_attribute("parse_error", True)
        span.set_attribute(S.OUTPUT_VALUE, json.dumps(result))
        span.set_attribute(S.OUTPUT_MIME_TYPE, "application/json")
        return result


async def triage(message: str, history: list[dict]) -> dict:
    hist = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-6:]) or "(none)"
    out = await _structured_call(
        "triage", TRIAGE_TEMPLATE, TRIAGE_VERSION,
        {"history": hist, "message": message, "intents": json.dumps(INTENTS)},
        fallback={"intent": "other", "sentiment": "neutral", "urgency": "normal", "language": "en",
                  "injection_suspected": False, "summary": ""},
        schema=TRIAGE_SCHEMA,
    )
    if out.get("intent") not in INTENTS:
        out["intent"] = "other"
    return out


async def crm_note(message: str, triage_result: dict, tool_calls: list[dict], response: str) -> dict:
    actions = "\n".join(
        f"- {t['tool']}({json.dumps(t['args'])}) -> {'OK' if 'result' in t else 'ERROR: ' + str(t.get('error'))}"
        for t in tool_calls) or "(none)"
    return await _structured_call(
        "crm_note", CRM_TEMPLATE, CRM_VERSION,
        {"message": message, "triage": json.dumps(triage_result), "actions": actions, "response": response},
        fallback={"summary": "", "resolution": "pending_customer", "follow_up": None, "tags": []},
        schema=CRM_SCHEMA,
    )
