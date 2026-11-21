"""Online evals: score unlabeled production traces and write the scores back to Phoenix.

    uv run python -m scripts.eval_production                 # evaluate traces from the last 24h
    uv run python -m scripts.eval_production --hours 1

Flow:
  1. Pull spans for the prod project from Phoenix.
  2. Rebuild each agent turn (root AGENT span + its TOOL child spans).
  3. Run reference-free evaluators (no golden answer exists in prod):
       groundedness (LLM), resolution (LLM), tone (LLM), privacy_guard (code), tool_errors (code)
  4. Log them as span annotations -> they show up as columns/filters on the traces in Phoenix,
     e.g. filter `annotations['groundedness'].label == 'hallucinated'` to find bad conversations.
"""

import argparse
import asyncio
import json
from datetime import datetime, timedelta, timezone

import pandas as pd
from phoenix.client import Client

from evals.evaluators import privacy_guard
from evals.judges import judge_groundedness, judge_resolution, judge_tone
from support_agent.config import PHOENIX_BASE_URL, PROD_PROJECT, PROVIDERS, set_provider, setup_tracing


def _loads(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v


def build_turns(spans: pd.DataFrame) -> list[dict]:
    if "context.span_id" not in spans.columns:
        spans = spans.reset_index()
    else:
        spans = spans.reset_index(drop=True)
    roots = spans[spans["name"] == "support_agent"]
    tools = spans[spans["span_kind"] == "TOOL"].sort_values("start_time")
    turns = []
    for _, r in roots.iterrows():
        # TOOL spans live under react_loop, so group by trace instead of direct parent
        children = tools[tools["context.trace_id"] == r["context.trace_id"]]
        calls = []
        for _, t in children.iterrows():
            out = _loads(t.get("attributes.output.value"))
            call = {"tool": t["name"], "args": _loads(t.get("attributes.input.value")) or {}}
            if isinstance(out, dict) and set(out) == {"error"}:
                call["error"] = out["error"]
            else:
                call["result"] = out
            calls.append(call)
        turns.append({
            "span_id": r["context.span_id"],
            "customer_id": r.get("attributes.user.id"),
            "message": r.get("attributes.input.value") or "",
            "response": r.get("attributes.output.value") or "",
            "tool_calls": calls,
            "status": r.get("status_code"),
        })
    return turns


async def evaluate_turn(sem, turn) -> list[dict]:
    async with sem:
        g, res, tone = await asyncio.gather(
            judge_groundedness(turn["message"], turn["response"], turn["tool_calls"], turn["customer_id"]),
            judge_resolution(turn["message"], turn["response"], turn["tool_calls"]),
            judge_tone(turn["message"], turn["response"]),
        )
    priv = privacy_guard({"customer_id": turn["customer_id"]}, {"response": turn["response"]}, {})
    errs = [c for c in turn["tool_calls"] if "error" in c]
    tool_err = {
        "score": 0.0 if errs else 1.0,
        "label": "has_errors" if errs else "clean",
        "explanation": "; ".join(f"{c['tool']}: {c['error']}" for c in errs) or "No tool errors.",
    }
    rows = []
    for name, r, kind in [("groundedness", g, "LLM"), ("resolution", res, "LLM"), ("tone", tone, "LLM"),
                          ("privacy_guard", priv, "CODE"), ("tool_errors", tool_err, "CODE")]:
        rows.append({"span_id": turn["span_id"], "name": name, "annotator_kind": kind,
                     "label": r["label"], "score": r["score"], "explanation": r["explanation"]})
    return rows


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--provider", choices=list(PROVIDERS), default=None, help="provider for the LLM judges")
    args = ap.parse_args()
    if args.provider:
        set_provider(args.provider)
    setup_tracing(project_name="shopnova-online-evals")  # judge LLM calls get their own project

    client = Client(base_url=PHOENIX_BASE_URL)
    start = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    spans = client.spans.get_spans_dataframe(project_name=PROD_PROJECT, start_time=start, limit=10000, timeout=60)
    if spans.empty:
        print("No spans found — run `python -m scripts.simulate_traffic` first.")
        return
    turns = build_turns(spans)
    print(f"Evaluating {len(turns)} agent turns from '{PROD_PROJECT}'...")

    sem = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*[evaluate_turn(sem, t) for t in turns], return_exceptions=True)
    rows = [row for r in results if not isinstance(r, Exception) for row in r]
    failed = [r for r in results if isinstance(r, Exception)]
    if failed:
        print(f"{len(failed)} turns failed to evaluate, e.g. {failed[0]!r}")

    df = pd.DataFrame(rows)
    client.spans.log_span_annotations_dataframe(dataframe=df, sync=True)

    summary = df.groupby("name")["score"].mean().round(2)
    print("\nMean scores on production traffic:")
    print(summary.to_string())
    bad = df[(df["score"] < 1) & df["name"].isin(["groundedness", "resolution", "privacy_guard"])]
    if not bad.empty:
        print(f"\nFlagged ({len(bad)}):")
        for _, b in bad.iterrows():
            print(f"  span {b['span_id']}  {b['name']}={b['label']}: {str(b['explanation'])[:150]}")
    print(f"\nAnnotations logged. Open {PHOENIX_BASE_URL} -> Projects -> {PROD_PROJECT}")


if __name__ == "__main__":
    asyncio.run(main())
