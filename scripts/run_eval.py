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

from da_verify.agent import CONDITIONS
from da_verify.eval.metrics import TaskScore, aggregate
from da_verify.eval.scoring import score_response
from da_verify.llm import LLMClient
from da_verify.sandbox import KernelSandbox
from da_verify.tasks.loader import load_tasks, tasks_by_id
from da_verify.tasks.sampler import load_subset_ids

ROOT = Path(__file__).resolve().parents[1]
SUBSET = ROOT / "data" / "subsets" / "headline_40.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--condition", choices=list(CONDITIONS), default="c0",
                    help="verification condition: c0 (none) | c1 (self-verify)")
    ap.add_argument("--k", type=int, default=1, help="samples per task (pass@k)")
    ap.add_argument("--temp", type=float, default=None, help="default 0 if k==1 else 0.7")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel tasks (default 1=sequential). API I/O-bound; "
                         "raise to speed up the FIRST run. Watch provider rate limits.")
    ap.add_argument("--verifier-model", default=None,
                    help="C2 only: use a DIFFERENT model for the verifier (e.g. MiniMax-M3)")
    args = ap.parse_args()
    temp = args.temp if args.temp is not None else (0.0 if args.k == 1 else 0.7)

    tasks = tasks_by_id(load_tasks())
    ids = load_subset_ids(SUBSET)[: args.n]
    llm = LLMClient.from_env(temperature=temp)
    run_fn = CONDITIONS[args.condition]
    verifier_llm = None
    vtag = ""
    if args.condition == "c2" and args.verifier_model:
        verifier_llm = LLMClient.from_env(temperature=temp, model=args.verifier_model)
        vtag = f"_v{args.verifier_model.replace('/', '_')}"
    print(f"model={llm.model} condition={args.condition}"
          f"{' verifier='+verifier_llm.model if verifier_llm else ''} temp={temp} n={len(ids)} k={args.k}\n")

    stem = f"eval_{llm.model.replace('/', '_')}_{args.condition}{vtag}_n{len(ids)}_k{args.k}"
    out_jsonl = ROOT / "results" / f"{stem}.jsonl"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    def run_one(i):
        """Run all k samples for one task; return (id, TaskScore, jsonl_row).
        Self-contained so it's safe to run concurrently (own sandbox per sample,
        atomic cache writes, separate kernel process)."""
        t = tasks[i]
        samples = []
        for s in range(args.k):
            with KernelSandbox(data_csv=t.table_path) as sb:
                if verifier_llm is not None:
                    tr = run_fn(t, llm, sb, max_steps=args.max_steps, sample_id=s,
                                verifier_llm=verifier_llm)
                else:
                    tr = run_fn(t, llm, sb, max_steps=args.max_steps, sample_id=s)
            samples.append(score_response(t, tr.final_response))
        n_correct = sum(x.correct for x in samples)
        sc = TaskScore(
            id=t.id, level=t.level, n_samples=args.k, n_correct=n_correct,
            n_format_ok=sum(x.format_ok for x in samples),
            n_candidate=sum(x.candidate for x in samples),
        )
        row = {
            "id": t.id, "level": t.level, "n_correct": n_correct, "k": args.k,
            "n_lenient": sum(x.lenient_correct for x in samples),
            "pass@1": round(sc.pass_at_1, 3),
            "samples": [{"correct": x.correct, "lenient_correct": x.lenient_correct,
                         "format_ok": x.format_ok, "candidate": x.candidate,
                         "predicted": x.predicted} for x in samples],
            "gold": {g.name: g.value for g in t.gold},
        }
        return i, sc, row

    results: dict[int, tuple] = {}
    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, i): i for i in ids}
            for fut in as_completed(futs):
                i, sc, row = fut.result()
                results[i] = (sc, row)
                print(f"  id={i:<4} {row['level']:<6} correct={row['n_correct']}/{args.k} (done)")
    else:
        for i in ids:
            _, sc, row = run_one(i)
            results[i] = (sc, row)
            print(f"  id={i:<4} {row['level']:<6} correct={row['n_correct']}/{args.k}  pass@1={sc.pass_at_1:.2f}")

    # Write in subset order (deterministic file regardless of completion order).
    scores: list[TaskScore] = []
    with out_jsonl.open("w", encoding="utf-8") as f:
        for i in ids:
            sc, row = results[i]
            scores.append(sc)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = aggregate(scores, k=args.k)
    (ROOT / "results" / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    total_samples = len(results) * args.k
    lenient_p1 = sum(r["n_lenient"] for _, r in results.values()) / total_samples if total_samples else 0.0

    print("\n" + "=" * 56)
    print(f"EVAL  model={llm.model}  condition={args.condition}  n={summary['n_tasks']}  k={args.k}  temp={temp}")
    print(f"  pass@1 (headline):        {summary['pass@1']:.1%}")
    print(f"  pass@1 (format-forgiving):{lenient_p1:.1%}  "
          f"(+{lenient_p1 - summary['pass@1']:.1%} = single-field @name mismatches, NOT in headline)")
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
