.PHONY: phoenix setup chat chat-librechat smoke exp-v1 exp-v2 exp-librechat traffic eval-prod demo sync-dataset

phoenix:          ## start Phoenix UI + collector on http://localhost:6006
	uv run phoenix serve

setup:            ## register model pricing + prompt versions in Phoenix (run once)
	uv run python -m scripts.setup_phoenix

chat:             ## talk to the agent (gpt-6.2-sol via OpenRouter, native tool calls), traced
	uv run python -m scripts.chat --provider openrouter --customer C005 --debug

chat-librechat:   ## same agent on the LibreChat endpoint (JSON action protocol) — fallback
	uv run python -m scripts.chat --provider librechat --customer C005 --debug

smoke:            ## fast, cheap: code evals only on 5 random cases (not recorded)
	uv run python -m evals.run_experiment --provider openrouter --prompt v2 --code-only --limit 5

exp-v1:           ## offline experiment: gpt-6.2-sol, baseline prompt
	uv run python -m evals.run_experiment --provider openrouter --prompt v1 --name "sol-v1-baseline"

exp-v2:           ## offline experiment: gpt-6.2-sol, SOP prompt
	uv run python -m evals.run_experiment --provider openrouter --prompt v2 --name "sol-v2-sop"

exp-librechat:    ## the same two experiments on the LibreChat endpoint
	uv run python -m evals.run_experiment --provider librechat --prompt v1 --name "librechat-v1-baseline"
	uv run python -m evals.run_experiment --provider librechat --prompt v2 --name "librechat-v2-sop"

traffic:          ## simulate production conversations (mix of v1/v2)
	uv run python -m scripts.simulate_traffic --provider openrouter --mix

eval-prod:        ## online evals on production traces -> span annotations
	uv run python -m scripts.eval_production --provider openrouter

sync-dataset:     ## re-upload evals/dataset.py to Phoenix (DELETES the dataset's experiments)
	uv run python -m evals.run_experiment --provider openrouter --prompt v2 --code-only --recreate-dataset --name "dataset-sync check"

demo: exp-v1 exp-v2 traffic eval-prod   ## (run `make setup` once beforehand)
