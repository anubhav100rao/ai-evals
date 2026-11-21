"""Chat with the ShopNova support agent in the terminal. Every turn is traced to Phoenix.

    uv run python -m scripts.chat --customer C001
    uv run python -m scripts.chat --customer C002 --prompt v1 --once "I want a refund for ORD-1005"
"""

import argparse
import asyncio
import json
import uuid

from support_agent.agent import assistant_turn, run_agent
from support_agent.config import PROVIDERS, flush_traces, provider, set_provider, setup_tracing


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--customer", default="C001")
    ap.add_argument("--prompt", default="v2", choices=["v1", "v2"])
    ap.add_argument("--once", help="send a single message and exit")
    ap.add_argument("--debug", action="store_true", help="print tool calls")
    ap.add_argument("--provider", choices=list(PROVIDERS), default=None)
    args = ap.parse_args()
    if args.provider:
        set_provider(args.provider)

    setup_tracing()
    session_id = f"cli-{uuid.uuid4().hex[:8]}"
    history: list[dict] = []
    print(f"ShopNova support (customer {args.customer}, prompt {args.prompt}, "
          f"{provider().name}:{provider().agent_model}, session {session_id})")

    while True:
        msg = args.once or input("\nyou> ").strip()
        if not msg or msg in {"exit", "quit"}:
            break
        out = await run_agent(msg, args.customer, history=history, prompt_version=args.prompt, session_id=session_id)
        if args.debug or args.once:
            for t in out["tool_calls"]:
                res = t.get("result", {"error": t.get("error")})
                print(f"  [tool] {t['tool']}({json.dumps(t['args'])}) -> {json.dumps(res, default=str)[:200]}")
        print(f"nova> {out['response']}")
        print(f"  ({out['steps']} steps, {out['latency_s']}s, trace {out['trace_id']})")
        history += [{"role": "user", "content": msg}, assistant_turn(out["response"])]
        if args.once:
            break
    flush_traces()


if __name__ == "__main__":
    asyncio.run(main())
