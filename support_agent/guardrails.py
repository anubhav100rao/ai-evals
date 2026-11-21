"""Input / output guardrails, traced as OpenInference GUARDRAIL spans.

input_guardrail   redacts payment data / secrets before they reach the LLM and flags likely
                  prompt-injection attempts (the agent is warned, not blocked).
output_guardrail  blocks replies that leak another customer's PII or internal data, replacing
                  them with a safe fallback. The block is recorded on the span so you can
                  filter for it in Phoenix.
"""

from __future__ import annotations

import json
import re

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes as S
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from .tools import SEED_CUSTOMERS, SEED_ORDERS

_tracer = trace.get_tracer("shopnova.guardrails")

REDACTIONS = [
    ("card_number", re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[REDACTED_CARD]"),
    ("cvv", re.compile(r"\b(cvv|cvc|security code)\s*(is|:)?\s*\d{3,4}\b", re.I), "[REDACTED_CVV]"),
    ("password", re.compile(r"\b(password|passcode|pwd)\s*(is|:)\s*\S+", re.I), "[REDACTED_PASSWORD]"),
]
INJECTION_PATTERNS = [
    r"ignore (all |any )?(your |the )?(previous|prior|above)? ?(instructions|rules)",
    r"ignore your rules", r"system override", r"admin mode", r"developer mode", r"you are now",
    r"\bDAN\b", r"jailbreak", r"reveal (your )?(system )?prompt", r"pretend (to be|you are)",
]
SAFE_FALLBACK = ("I'm sorry, I can't share that information. I can only help with orders and details on your own "
                 "ShopNova account — is there anything about your orders I can help with?")


def input_guardrail(message: str) -> dict:
    with _tracer.start_as_current_span("input_guardrail") as span:
        span.set_attribute(S.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.GUARDRAIL.value)
        span.set_attribute(S.INPUT_VALUE, message)
        sanitized, redactions = message, []
        for kind, rx, repl in REDACTIONS:
            if rx.search(sanitized):
                # don't treat order numbers / tracking numbers as card numbers
                if kind == "card_number" and not re.search(r"\b(?:\d[ -]?){15,19}\b", sanitized):
                    continue
                sanitized = rx.sub(repl, sanitized)
                redactions.append(kind)
        injections = [p for p in INJECTION_PATTERNS if re.search(p, message, re.I)]
        result = {
            "action": "allow",
            "sanitized_message": sanitized,
            "redactions": redactions,
            "injection_suspected": bool(injections),
            "matched_patterns": injections,
        }
        span.set_attribute("guardrail.name", "input_pii_and_injection")
        span.set_attribute("guardrail.action", "redact" if redactions else "allow")
        span.set_attribute("guardrail.injection_suspected", bool(injections))
        span.set_attribute(S.OUTPUT_VALUE, json.dumps(result))
        span.set_attribute(S.OUTPUT_MIME_TYPE, "application/json")
        return result


def foreign_pii(customer_id: str) -> list[str]:
    """PII belonging to every customer except the one in this chat."""
    pii: list[str] = []
    for c in SEED_CUSTOMERS:
        if c["customer_id"] == customer_id:
            continue
        parts = c["phone"].split("-")
        pii += [c["name"], c["email"], f"{parts[-2]}-{parts[-1]}"]
    for o in SEED_ORDERS:
        if o["customer_id"] == customer_id:
            continue
        pii += [o["shipping_address"].split(",")[0], o.get("tracking_number", "")]
    return [p for p in pii if p]


INTERNAL_MARKERS = ["TOOL_RESULT", '"type": "tool_call"', "customer_id\":", "Standard operating procedure"]


def output_guardrail(response: str, customer_id: str) -> dict:
    with _tracer.start_as_current_span("output_guardrail") as span:
        span.set_attribute(S.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.GUARDRAIL.value)
        span.set_attribute(S.INPUT_VALUE, response)
        low = response.lower()
        leaks = sorted({p for p in foreign_pii(customer_id) if p.lower() in low})
        internal = [m for m in INTERNAL_MARKERS if m.lower() in low]
        blocked = bool(leaks or internal)
        result = {
            "action": "block" if blocked else "allow",
            "pii_leaks": leaks,
            "internal_leaks": internal,
            "final_response": SAFE_FALLBACK if blocked else response,
        }
        span.set_attribute("guardrail.name", "output_privacy")
        span.set_attribute("guardrail.action", result["action"])
        if blocked:
            span.set_status(Status(StatusCode.ERROR, f"blocked: pii={leaks} internal={internal}"))
        span.set_attribute(S.OUTPUT_VALUE, json.dumps(result))
        span.set_attribute(S.OUTPUT_MIME_TYPE, "application/json")
        return result
