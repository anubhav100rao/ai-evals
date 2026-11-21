"""Run the golden dataset through the agent as a Phoenix experiment.

    uv run python -m evals.run_experiment --prompt v1            # baseline
    uv run python -m evals.run_experiment --prompt v2            # improved prompt
    uv run python -m evals.run_experiment --prompt v2 --code-only --limit 5   # fast smoke run

Phoenix stores: the dataset (versioned), each run's output, its full agent trace
(AGENT -> LLM / TOOL spans), and every evaluator's score + explanation. Compare
experiments side by side under Datasets -> shopnova-support-golden -> Experiments.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import httpx
from phoenix.client import AsyncClient

from support_agent.agent import run_agent
from support_agent.config import PHOENIX_BASE_URL, PROVIDERS, flush_traces, provider, set_provider, setup_tracing

from .dataset import DATASET_NAME, GOLDEN_CASES
from .evaluators import ALL_EVALUATORS, CODE_EVALUATORS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


async def get_or_create_dataset(client: AsyncClient, recreate: bool = False):
    try:
        ds = await client.datasets.get_dataset(dataset=DATASET_NAME)
    except Exception:
        ds = None
    if ds is not None:
        remote_ids = {e["metadata"].get("case_id") for e in ds.examples}
        local_ids = {c["metadata"]["case_id"] for c in GOLDEN_CASES}
        if recreate:
            async with httpx.AsyncClient(base_url=PHOENIX_BASE_URL) as h:
                (await h.delete(f"/v1/datasets/{ds.id}")).raise_for_status()
            print(f"Deleted dataset '{DATASET_NAME}' (and its experiments) to recreate it")
        else:
            if remote_ids != local_ids:
                print(f"WARNING: Phoenix dataset differs from evals/dataset.py "
                      f"(+{len(local_ids - remote_ids)} / -{len(remote_ids - local_ids)} cases). "
                      "Re-run with --recreate-dataset to sync.")
            print(f"Using existing dataset '{DATASET_NAME}' ({len(ds)} examples)")
            return ds
    ds = await client.datasets.create_dataset(
        name=DATASET_NAME,
        inputs=[c["input"] for c in GOLDEN_CASES],
        outputs=[c["expected"] for c in GOLDEN_CASES],
        metadata=[c["metadata"] for c in GOLDEN_CASES],
        dataset_description="Golden test cases for the ShopNova customer-support agent "
                            "(order status, returns/refunds, order changes, policy, safety/privacy, escalation).",
    )
    print(f"Created dataset '{DATASET_NAME}' with {len(GOLDEN_CASES)} examples")
    return ds


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="v2", choices=["v1", "v2"])
    ap.add_argument("--code-only", action="store_true", help="skip LLM-as-judge evaluators")
    ap.add_argument("--limit", type=int, default=0, help="dry-run on N random examples (not recorded)")
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--name", default=None)
    ap.add_argument("--recreate-dataset", action="store_true",
                    help="delete + re-upload the golden dataset (drops its experiments)")
    ap.add_argument("--no-guardrails", action="store_true", help="disable input/output guardrails")
    ap.add_argument("--provider", choices=list(PROVIDERS), default=None, help="LLM provider (default: LLM_PROVIDER)")
    args = ap.parse_args()
    if args.provider:
        set_provider(args.provider)
    p = provider()

    setup_tracing(project_name="shopnova-support-experiments")
    client = AsyncClient(base_url=PHOENIX_BASE_URL)
    dataset = await get_or_create_dataset(client, recreate=args.recreate_dataset)

    async def task(input: dict, metadata: dict) -> dict:
        return await run_agent(
            input["message"],
            input["customer_id"],
            prompt_version=args.prompt,
            session_id=f"exp-{args.prompt}-{metadata.get('case_id')}",
            guardrails=not args.no_guardrails,
        )

    evaluators = CODE_EVALUATORS if args.code_only else ALL_EVALUATORS
    stamp = datetime.now().strftime("%m%d-%H%M")
    name = args.name or f"{p.name}-prompt-{args.prompt} {stamp}"

    ran = await client.experiments.run_experiment(
        dataset=dataset,
        task=task,
        evaluators=evaluators,
        experiment_name=name,
        experiment_description=f"Support agent, system prompt {args.prompt}, {p.agent_model} ({p.name})",
        experiment_metadata={"prompt_version": args.prompt, "provider": p.name, "agent_model": p.agent_model,
                             "judge_model": p.judge_model, "tool_mode": "native" if p.native_tools else "json_protocol",
                             "guardrails": not args.no_guardrails},
        concurrency=args.concurrency,
        timeout=300,
        dry_run=args.limit or False,
        print_summary=False,
    )
    flush_traces()
    summarize(ran, name, dataset)


def summarize(ran, name: str, dataset) -> None:
    """Print a per-evaluator and per-category scoreboard, and save raw results."""
    runs = {r["id"]: r for r in ran["task_runs"]}
    examples = {e["id"]: e for e in dataset.examples}
    by_eval: dict[str, list[float]] = defaultdict(list)
    by_cat: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    failures = []
    errors = [ev for ev in ran["evaluation_runs"] if ev.error]
    if errors:
        print(f"\n{len(errors)} evaluator errors, e.g. {errors[0].name}: {errors[0].error[:200]}")
    for ev in ran["evaluation_runs"]:
        res = ev.result or {}
        score = res.get("score")
        if score is None:
            continue
        run = runs.get(ev.experiment_run_id)
        ex = examples.get(run["dataset_example_id"]) if run else None
        cat = (ex or {}).get("metadata", {}).get("category", "?")
        by_eval[ev.name].append(score)
        by_cat[cat][ev.name].append(score)
        if score < 1 and ev.name not in ("tone",):
            failures.append({
                "case": (ex or {}).get("metadata", {}).get("case_id"),
                "evaluator": ev.name,
                "label": res.get("label"),
                "explanation": res.get("explanation"),
            })

    print(f"\n=== {name} ===")
    print(f"{'evaluator':<16} {'mean':>6}  n")
    for k, v in by_eval.items():
        print(f"{k:<16} {sum(v) / len(v):>6.2f}  {len(v)}")

    evals = list(by_eval)
    print(f"\n{'category':<16} " + " ".join(f"{e[:10]:>10}" for e in evals))
    for cat, scores in sorted(by_cat.items()):
        cells = [f"{sum(scores[e]) / len(scores[e]):>10.2f}" if scores.get(e) else f"{'-':>10}" for e in evals]
        print(f"{cat:<16} " + " ".join(cells))

    if failures:
        print(f"\nFailures ({len(failures)}):")
        for f in sorted(failures, key=lambda x: (x["case"] or "", x["evaluator"]))[:40]:
            print(f"  [{f['case']}] {f['evaluator']}={f['label']}: {str(f['explanation'])[:160]}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{name.replace(' ', '_')}.json"
    out.write_text(json.dumps({
        "experiment": name,
        "experiment_id": ran.get("experiment_id"),
        "means": {k: sum(v) / len(v) for k, v in by_eval.items()},
        "by_category": {c: {e: sum(s) / len(s) for e, s in d.items()} for c, d in by_cat.items()},
        "failures": failures,
    }, indent=2, default=str))
    print(f"\nSaved {out}")
    print(f"Open Phoenix: {PHOENIX_BASE_URL}  ->  Datasets -> {DATASET_NAME} -> Experiments")


if __name__ == "__main__":
    asyncio.run(main())
