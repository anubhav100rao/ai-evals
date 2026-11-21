"""One-time Phoenix setup (pricing is skipped if present; re-running adds new prompt versions):

1. Model pricing   -> Phoenix computes $ cost per LLM span / trace / project. openai/gpt-6.2-sol is
                      priced by Phoenix's built-in table; the LibreChat agent model gets ILLUSTRATIVE
                      demo pricing here.
2. Prompt registry -> every prompt the system uses is versioned in Phoenix > Prompts,
                      with tags (production / baseline) so the UI shows which version is live.

    uv run python -m scripts.setup_phoenix
"""

import httpx
from phoenix.client import Client
from phoenix.client.types import PromptVersion

from evals.judges import CORRECTNESS_TEMPLATE, GROUNDEDNESS_TEMPLATE, JUDGE_SYSTEM, RESOLUTION_TEMPLATE, TONE_TEMPLATE
from support_agent.config import PHOENIX_BASE_URL, provider
from support_agent.prompts import prompt_template
from support_agent.tools import openai_tools
from support_agent.side_llms import CRM_TEMPLATE, TRIAGE_TEMPLATE

# Illustrative pricing ($ per 1M tokens) for the demo model.
PRICING = {"name": "gpt-6-sol (demo pricing)", "pattern": r"(?i)^agent__.*", "prompt": 1.25, "completion": 10.0}

CREATE_MODEL = """
mutation($input: CreateModelMutationInput!) { createModel(input: $input) { model { id name } } }"""
LIST_MODELS = "{ generativeModels(first: 2000) { edges { node { name namePattern } } } }"


def setup_pricing() -> None:
    gql = f"{PHOENIX_BASE_URL}/graphql"
    try:
        existing = httpx.post(gql, json={"query": LIST_MODELS}, timeout=10).json()
        names = {e["node"]["name"] for e in existing["data"]["generativeModels"]["edges"]}
    except Exception:
        names = set()
    if PRICING["name"] in names:
        print(f"pricing: '{PRICING['name']}' already registered")
        return
    variables = {"input": {
        "name": PRICING["name"], "provider": "openai", "namePattern": PRICING["pattern"], "modelType": "LLM",
        "costs": [
            {"tokenType": "input", "kind": "PROMPT", "costPerMillionTokens": PRICING["prompt"]},
            {"tokenType": "output", "kind": "COMPLETION", "costPerMillionTokens": PRICING["completion"]},
        ],
    }}
    r = httpx.post(gql, json={"query": CREATE_MODEL, "variables": variables}, timeout=10).json()
    if r.get("errors"):
        print("pricing: failed ->", r["errors"][0]["message"])
    else:
        print(f"pricing: registered {PRICING['name']} for model pattern {PRICING['pattern']}")


def setup_prompts() -> None:
    client = Client(base_url=PHOENIX_BASE_URL)
    model = provider().agent_model
    # Agent prompts are stored WITH their 16 tool schemas (native function-calling version).
    for version, desc, tag in [("v1", "v1 naive baseline", "baseline"), ("v2", "v2 SOP with guardrails", "production")]:
        pv = PromptVersion.from_openai(
            {"model": model, "messages": [{"role": "system", "content": prompt_template(version, native_tools=True)}],
             "tools": openai_tools(), "tool_choice": "auto"},
            template_format="F_STRING", description=desc)
        _create(client, "shopnova-support-agent", pv, desc, tag)
    others = [
        ("shopnova-triage", "intent/sentiment/urgency classifier", [("user", TRIAGE_TEMPLATE)], "production"),
        ("shopnova-crm-note", "structured CRM note writer", [("user", CRM_TEMPLATE)], "production"),
        ("judge-correctness", "LLM judge vs reference answer", [("system", JUDGE_SYSTEM), ("user", CORRECTNESS_TEMPLATE)], None),
        ("judge-groundedness", "hallucination judge", [("system", JUDGE_SYSTEM), ("user", GROUNDEDNESS_TEMPLATE)], None),
        ("judge-tone", "1-5 communication quality", [("system", JUDGE_SYSTEM), ("user", TONE_TEMPLATE)], None),
        ("judge-resolution", "reference-free resolution judge", [("system", JUDGE_SYSTEM), ("user", RESOLUTION_TEMPLATE)], None),
    ]
    for name, desc, msgs, tag in others:
        pv = PromptVersion([{"role": r, "content": c} for r, c in msgs], model_name=model, description=desc,
                           template_format="F_STRING")
        _create(client, name, pv, desc, tag)


def _create(client: Client, name: str, version: PromptVersion, desc: str, tag: str | None) -> None:
    created = client.prompts.create(name=name, version=version, prompt_description=desc)
    if tag:
        try:
            client.prompts.tags.create(prompt_version_id=created.id, name=tag)
        except Exception as e:
            print(f"  tag '{tag}' on {name}: {e}")
    print(f"prompt: {name} <- {desc}" + (f" [{tag}]" if tag else ""))


if __name__ == "__main__":
    setup_pricing()
    setup_prompts()
