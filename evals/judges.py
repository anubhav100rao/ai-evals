"""LLM-as-judge evaluators.

The chat endpoint has no native structured output / function calling, so judges ask for a single
JSON object ({"explanation": ..., "label": ...}) and we parse it ourselves. Explanation comes
first so the judge reasons before committing to a label.
"""

from __future__ import annotations

import json
import re

from support_agent.config import chat, provider
from support_agent.tools import POLICIES, SEED_CUSTOMERS, TODAY

JUDGE_SYSTEM = (
    "You are a meticulous QA reviewer for a customer-support AI. You grade one conversation turn at a time. "
    "Do not run code or use tools. Reply with exactly one JSON object and nothing else."
)

CORRECTNESS_TEMPLATE = """\
Grade whether the AGENT REPLY resolves the customer's request as well as the REFERENCE ANSWER does.

[CUSTOMER MESSAGE]
{message}

[REFERENCE ANSWER — what a perfect agent would convey]
{reference}

[AGENT REPLY]
{response}

[ACTIONS THE AGENT TOOK]
{actions}

Rules:
- Judge substance, not wording. Extra helpful detail is fine if it does not contradict the reference.
- Consider actions: e.g. if the reference says a refund is issued, the agent must have actually issued it.
- "correct": same outcome and all key facts. "partially_correct": right direction but missing or vague on a
  key fact/action. "incorrect": wrong outcome, wrong facts, or a policy-violating action.

Reply as JSON: {{"explanation": "<2-3 sentences>", "label": "correct" | "partially_correct" | "incorrect"}}"""

GROUNDEDNESS_TEMPLATE = """\
Check whether every factual claim in the AGENT REPLY is supported by the EVIDENCE (tool results the agent
received + the company policy knowledge base). Today's date is {today}.

[AGENT CONTEXT — known to the agent, counts as evidence]
The agent is "Nova", ShopNova's automated customer-support assistant. It only handles ShopNova support topics,
can create specialist tickets (escalate_to_human), and is chatting with the authenticated customer {customer_name}.

[CUSTOMER MESSAGE]
{message}

[EVIDENCE: TOOL CALLS THE AGENT MADE AND THEIR RESULTS]
{tool_results}

[EVIDENCE: POLICY KNOWLEDGE BASE]
{policies}

[AGENT REPLY]
{response}

A claim is a hallucination if it states a specific fact (status, date, amount, tracking number, ticket/refund id,
policy rule, timeline, promise) that is not in, or contradicts, the evidence. Check date arithmetic against
today's date. NOT hallucinations: greetings, empathy, offers to help, asking the customer for information,
declining a request, and statements about what the agent itself did or did not do in this conversation that are
consistent with the tool-call list (e.g. "I haven't issued a refund" when no issue_refund call exists),
reasonable paraphrases or direct implications of a policy (e.g. "I can't apply a discount" from "agents cannot
create discount codes"), and generic soft wording ("the quickest way", "a specialist will follow up with next
steps"). Only flag concrete, checkable facts that are wrong or invented.

Reply as JSON: {{"explanation": "<list any unsupported claims, or say none>", "label": "grounded" | "hallucinated"}}"""

TONE_TEMPLATE = """\
Rate the customer-service quality of the AGENT REPLY on a 1-5 scale.

[CUSTOMER MESSAGE]
{message}

[AGENT REPLY]
{response}

5 = warm, empathetic where appropriate, concise, clear next steps, professional.
4 = good, minor issues (slightly long, a bit generic).
3 = acceptable but robotic, verbose, or misses the customer's emotion.
2 = unclear, curt, or confusing.
1 = rude, dismissive, or unprofessional.
Do not judge factual correctness here, only communication quality.

Reply as JSON: {{"explanation": "<1-2 sentences>", "score": 1 | 2 | 3 | 4 | 5}}"""


RESOLUTION_TEMPLATE = """\
You are reviewing a production support conversation turn with NO reference answer.
Decide whether the agent moved the customer's issue to a correct resolution given the evidence.

[CUSTOMER MESSAGE]
{message}

[ACTIONS THE AGENT TOOK AND RESULTS]
{tool_results}

[COMPANY POLICY]
{policies}

[AGENT REPLY]
{response}

"resolved": request handled (answered, action completed, correctly declined per policy, or escalated with ticket).
"needs_followup": reasonable but the customer will likely have to come back (vague, asked for info the agent could
have looked up, missing next step).
"unresolved": wrong action, policy violation, ignored the request, or misleading answer.

Reply as JSON: {{"explanation": "<2 sentences>", "label": "resolved" | "needs_followup" | "unresolved"}}"""


async def judge_resolution(message: str, response: str, tool_calls: list[dict]) -> dict:
    out = await _ask_judge(RESOLUTION_TEMPLATE.format(
        message=message, tool_results=_tool_results_text(tool_calls), policies=_policies_text(), response=response),
        RESOLUTION_SCHEMA, "resolution")
    label = str(out.get("label", "")).strip().lower()
    score = {"resolved": 1.0, "needs_followup": 0.5, "unresolved": 0.0}.get(label, 0.0)
    return {"label": label, "score": score, "explanation": out.get("explanation", "")}


def _policies_text() -> str:
    return "\n".join(f"- {p['id']} {p['title']}: {p['body']}" for p in POLICIES)


def _tool_results_text(tool_calls: list[dict], limit: int = 6000) -> str:
    if not tool_calls:
        return "(no tools were called)"
    lines = []
    for t in tool_calls:
        res = t.get("result", {"error": t.get("error")})
        lines.append(f"- {t['tool']}({json.dumps(t['args'])}) -> {json.dumps(res, default=str)}")
    return "\n".join(lines)[:limit]


def _actions_text(tool_calls: list[dict]) -> str:
    if not tool_calls:
        return "(none)"
    return "\n".join(
        f"- {t['tool']}({json.dumps(t['args'])}) -> {'OK' if 'result' in t else 'ERROR: ' + str(t.get('error'))}"
        for t in tool_calls
    )


def _label_schema(labels: list[str]) -> dict:
    return {"type": "object", "additionalProperties": False, "required": ["explanation", "label"],
            "properties": {"explanation": {"type": "string"}, "label": {"type": "string", "enum": labels}}}


CORRECTNESS_SCHEMA = _label_schema(["correct", "partially_correct", "incorrect"])
GROUNDEDNESS_SCHEMA = _label_schema(["grounded", "hallucinated"])
RESOLUTION_SCHEMA = _label_schema(["resolved", "needs_followup", "unresolved"])
TONE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["explanation", "score"],
               "properties": {"explanation": {"type": "string"}, "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]}}}


async def _ask_judge(prompt: str, schema: dict | None = None, name: str = "judgement") -> dict:
    last_err = None
    for _ in range(3):
        text = await chat(
            [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
            provider().judge_model, json_schema=schema, schema_name=name,
        )
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError as e:
                last_err = e
        else:
            last_err = ValueError(f"judge returned no JSON: {text[:200]}")
    raise RuntimeError(f"judge failed to return JSON: {last_err}")


async def judge_correctness(message: str, response: str, reference: str, tool_calls: list[dict]) -> dict:
    out = await _ask_judge(CORRECTNESS_TEMPLATE.format(
        message=message, reference=reference, response=response, actions=_actions_text(tool_calls)),
        CORRECTNESS_SCHEMA, "correctness")
    label = str(out.get("label", "")).strip().lower()
    score = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}.get(label, 0.0)
    return {"label": label, "score": score, "explanation": out.get("explanation", "")}


async def judge_groundedness(message: str, response: str, tool_calls: list[dict], customer_id: str | None = None) -> dict:
    name = next((c["name"] for c in SEED_CUSTOMERS if c["customer_id"] == customer_id), "(unknown)")
    out = await _ask_judge(GROUNDEDNESS_TEMPLATE.format(
        today=TODAY.isoformat(), customer_name=name, message=message,
        tool_results=_tool_results_text(tool_calls), policies=_policies_text(), response=response),
        GROUNDEDNESS_SCHEMA, "groundedness")
    label = str(out.get("label", "")).strip().lower()
    return {"label": label, "score": 1.0 if label == "grounded" else 0.0, "explanation": out.get("explanation", "")}


async def judge_tone(message: str, response: str) -> dict:
    out = await _ask_judge(TONE_TEMPLATE.format(message=message, response=response), TONE_SCHEMA, "tone")
    try:
        raw = int(out.get("score"))
    except (TypeError, ValueError):
        raw = 1
    raw = min(max(raw, 1), 5)
    return {"label": f"{raw}/5", "score": (raw - 1) / 4, "explanation": out.get("explanation", "")}
