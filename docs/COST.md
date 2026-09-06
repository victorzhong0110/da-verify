# Cost narrative (internal — no invented dollars)

## 30-second interview answer
I do not publish a fake dollar total. I compare **relative LLM work**: C3 uses 2–3 solver runs and a programmatic gate (no LLM judge), so it is usually cheaper than C2's solver+verifier pair while scoring higher on the powered run. If you need $, sum `cache/llm` usage × the provider rate card and date it — recipe below.


Date: 2026-09-06
Rule: Do not claim a dollar total unless computed from recorded usage times a stated price table.

## What the repo contains
- results/*_summary.json — accuracy only (no usage or dollar fields).
- src/da_verify/llm/client.py — completions can store usage {prompt, completion} under cache/llm (local/gitignored; not rolled into public summaries).
- Headline study used MiniMax-M2.7 (M3 on one verifier arm). Provider invoices are not committed.

## Relative cost model (defendable without dollars)
| Condition | LLM work per task-sample | Notes |
| --- | --- | --- |
| C0 | ~1 solver trajectory | Baseline |
| C1 | solver + same-model self-check | Empirically flat at temp 0 |
| C2 | solver + independent verifier agent | More expensive; weaker than C3 at temp 0.7/k=5 |
| C3 | 2–3 solver runs + programmatic agreement | No LLM judge |

Interview line: C3 is cheaper aggregation than skeptical C2 because it drops the LLM verifier.

## How to produce a real dollar table later
1. Keep cache/llm after a run (or log usage into jsonl).
2. Sum prompt+completion tokens by condition.
3. Multiply by provider rates; record rates + date.
4. Optional: --max-spend once metering is in the summary path.

Until then, honest public statement: usage is recordable; published artifacts are not cost-accounted.
