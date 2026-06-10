"""Paired comparison of two condition runs (e.g. C0 vs C2).

- k = 1  -> per-task binary outcomes; Wilson CIs + exact McNemar, plus the list
           of tasks one condition fixed/broke.
- k > 1  -> per-task success RATES (n_correct/k); paired bootstrap 95% CI on the
           mean difference (McNemar no longer applies to rates).

Run: python3 scripts/compare_conditions.py \
       --a results/eval_<model>_c0_n40_k5.jsonl --a-name C0 \
       --b results/eval_<model>_c2_n40_k5.jsonl --b-name C2
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json

from da_verify.eval.stats import bootstrap_paired_diff, compare_paired


def _load(path: str) -> dict[int, tuple[int, int]]:
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    return {r["id"]: (r["n_correct"], r["k"]) for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    args = ap.parse_args()

    A, B = _load(args.a), _load(args.b)
    ids = sorted(set(A) & set(B))
    if set(A) != set(B):
        print(f"[warn] task sets differ; comparing {len(ids)} shared ids only")
    k = max(A[i][1] for i in ids)

    print("=" * 60)
    if k == 1:
        a = [A[i][0] > 0 for i in ids]
        b = [B[i][0] > 0 for i in ids]
        cmp = compare_paired(a, b)
        print(cmp.summary(args.a_name, args.b_name))
        print("-" * 60)
        print(f"{args.b_name} FIXED (A✗→B✓): {[i for i, ai, bi in zip(ids, a, b) if not ai and bi]}")
        print(f"{args.b_name} BROKE (A✓→B✗): {[i for i, ai, bi in zip(ids, a, b) if ai and not bi]}")
    else:
        a = [A[i][0] / A[i][1] for i in ids]
        b = [B[i][0] / B[i][1] for i in ids]
        rc = bootstrap_paired_diff(a, b)
        print(f"(k={k}, per-task success rates over {rc.n_tasks} tasks)")
        print(rc.summary(args.a_name, args.b_name))
        print("-" * 60)
        improved = [i for i in ids if B[i][0] / B[i][1] > A[i][0] / A[i][1]]
        worsened = [i for i in ids if B[i][0] / B[i][1] < A[i][0] / A[i][1]]
        print(f"{args.b_name} improved: {improved}")
        print(f"{args.b_name} worsened: {worsened}")


if __name__ == "__main__":
    main()
