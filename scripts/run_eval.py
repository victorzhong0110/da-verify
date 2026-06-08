"""W3 evaluation harness: run the C0 agent over the full subset, k samples/task,
score with the W1 verifier, report pass@1 / pass@k / reliability + by-level.

Run:  python3 scripts/run_eval.py --n 40 --k 1          # stable pass@1 (temp 0)
      python3 scripts/run_eval.py --n 10 --k 5 --temp 0.7  # pass@k slice
Out:  results/eval_<model>_n<n>_k<k>.jsonl  + results/eval_<...>_summary.json
Repro: LLM calls cached per (request, sample_id) -> deterministic re-runs.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from da_verify.agent import run_c0
from da_verify.eval.metrics import TaskScore, aggregate
from da_verify.llm import LLMClient
from da_verify.sandbox import KernelSandbox
from da_verify.tasks.loader import load_tasks, tasks_by_id
from da_verify.tasks.sampler import load_subset_ids
from da_verify.tasks.verifier import extract_answers, verify_response

ROOT = Path(__file__).resolve().parents[1]
SUBSET = ROOT / "data" / "subsets" / "headline_40.json"


def score_sample(task, response: str):
    vr = verify_response(task.id, response, [(g.name, g.value) for g in task.gold])
    extracted = extract_answers(response)
    required = [g.name for g in task.gold]
    return {
        "correct": vr.all_correct,
        "format_ok": all(r in extracted for r in required),
        "candidate": len(extracted) > 0,
        "predicted": vr.predicted,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--k", type=int, default=1, help="samples per task (pass@k)")
    ap.add_argument("--temp", type=float, default=None, help="default 0 if k==1 else 0.7")
    ap.add_argument("--max-steps", type=int, default=8)
    args = ap.parse_args()
    temp = args.temp if args.temp is not None else (0.0 if args.k == 1 else 0.7)

    tasks = tasks_by_id(load_tasks())
    ids = load_subset_ids(SUBSET)[: args.n]
    llm = LLMClient.from_env(temperature=temp)
    print(f"model={llm.model} temp={temp} n={len(ids)} k={args.k}\n")

    stem = f"eval_{llm.model.replace('/', '_')}_n{len(ids)}_k{args.k}"
    out_jsonl = ROOT / "results" / f"{stem}.jsonl"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    scores: list[TaskScore] = []
    with out_jsonl.open("w", encoding="utf-8") as f:
        for i in ids:
            t = tasks[i]
            samples = []
            for s in range(args.k):
                sb = KernelSandbox(data_csv=t.table_path)
                try:
                    sb.start()
                    tr = run_c0(t, llm, sb, max_steps=args.max_steps, sample_id=s)
                finally:
                    sb.shutdown()
                samples.append(score_sample(t, tr.final_response))
            n_correct = sum(x["correct"] for x in samples)
            sc = TaskScore(
                id=t.id, level=t.level, n_samples=args.k, n_correct=n_correct,
                n_format_ok=sum(x["format_ok"] for x in samples),
                n_candidate=sum(x["candidate"] for x in samples),
            )
            scores.append(sc)
            f.write(json.dumps({
                "id": t.id, "level": t.level, "n_correct": n_correct, "k": args.k,
                "pass@1": round(sc.pass_at_1, 3), "samples": samples,
                "gold": {g.name: g.value for g in t.gold},
            }, ensure_ascii=False) + "\n")
            print(f"  id={t.id:<4} {t.level:<6} correct={n_correct}/{args.k}  pass@1={sc.pass_at_1:.2f}")

    summary = aggregate(scores, k=args.k)
    (ROOT / "results" / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 56)
    print(f"EVAL  model={llm.model}  n={summary['n_tasks']}  k={args.k}  temp={temp}")
    print(f"  pass@1:               {summary['pass@1']:.1%}")
    if args.k > 1:
        print(f"  pass@{args.k}:               {summary[f'pass@{args.k}']:.1%}")
        print(f"  reliability(pass^{args.k}):   {summary['reliability(pass^k)']:.1%}")
    print(f"  format-ok rate:       {summary['format_ok_rate']:.1%}")
    print(f"  candidate rate:       {summary['candidate_rate']:.1%}")
    print("  by level: " + "  ".join(
        f"{lv}={d['pass@1']:.0%}" for lv, d in summary["by_level"].items()))
    print(f"  -> {out_jsonl.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
