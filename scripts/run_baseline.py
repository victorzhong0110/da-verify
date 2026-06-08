"""Run the C0 baseline over the first N headline tasks; score with the W1 verifier.

Run:  python3 scripts/run_baseline.py --n 10
Out:  results/baseline_c0.jsonl  +  printed summary
Repro: LLM responses are cached (cache/llm/), so re-runs are free & deterministic.

Reports three things on purpose:
  - accuracy            : the headline C0 number (verifier all_correct rate)
  - format-miss rate    : tasks where the model didn't emit the required @name[...]
                          (R2 — separates 'bad formatting' from 'wrong analysis')
  - candidate rate      : tasks where the model produced ANY answer field
                          (R3 — a low value = floor effect, model too weak to study)
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from da_verify.agent import run_c0
from da_verify.llm import LLMClient
from da_verify.sandbox import KernelSandbox
from da_verify.tasks.loader import load_tasks, tasks_by_id
from da_verify.tasks.sampler import load_subset_ids
from da_verify.tasks.verifier import extract_answers, verify_response

ROOT = Path(__file__).resolve().parents[1]
SUBSET = ROOT / "data" / "subsets" / "headline_40.json"
OUT = ROOT / "results" / "baseline_c0.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="how many of the headline subset to run")
    ap.add_argument("--max-steps", type=int, default=8)
    args = ap.parse_args()

    tasks = tasks_by_id(load_tasks())
    ids = load_subset_ids(SUBSET)[: args.n]
    llm = LLMClient.from_env()
    print(f"model={llm.model}  tasks={ids}\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with OUT.open("w", encoding="utf-8") as f:
        for i in ids:
            t = tasks[i]
            sb = KernelSandbox(data_csv=t.table_path)
            try:
                sb.start()
                trace = run_c0(t, llm, sb, max_steps=args.max_steps)
            finally:
                sb.shutdown()

            vr = verify_response(t.id, trace.final_response, [(g.name, g.value) for g in t.gold])
            extracted = extract_answers(trace.final_response)
            required = [g.name for g in t.gold]
            produced_candidate = len(extracted) > 0
            format_ok = all(r in extracted for r in required)

            row = {
                "id": t.id, "level": t.level, "csv": t.file_name,
                "all_correct": vr.all_correct,
                "n_correct_fields": vr.n_correct_fields, "n_fields": len(required),
                "produced_candidate": produced_candidate,
                "format_ok": format_ok,
                "steps": trace.steps, "tool_calls": trace.tool_calls,
                "hit_max_steps": trace.hit_max_steps, "error": trace.error,
                "predicted": vr.predicted, "gold": vr.gold,
                "final_response": trace.final_response[:500],
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            mark = "✓" if vr.all_correct else ("·" if format_ok else "✗fmt")
            err = f"  ERR={row['error']}" if row["error"] else ""
            print(f"  [{mark:4}] id={t.id:<4} {t.level:<6} steps={trace.steps} "
                  f"pred={vr.predicted} gold={vr.gold}{err}")

    n = len(rows)
    acc = sum(r["all_correct"] for r in rows) / n
    fmt_miss = sum(not r["format_ok"] for r in rows) / n
    cand = sum(r["produced_candidate"] for r in rows) / n
    errs = sum(bool(r["error"]) for r in rows)
    print("\n" + "=" * 56)
    print(f"C0 baseline  (model={llm.model}, n={n})")
    print(f"  accuracy (all_correct):  {acc:.1%}")
    print(f"  format-miss rate (R2):   {fmt_miss:.1%}")
    print(f"  candidate rate   (R3):   {cand:.1%}   (low => floor effect)")
    print(f"  API errors:              {errs}/{n}")
    print(f"  -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
