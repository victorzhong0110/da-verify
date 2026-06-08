"""Paired comparison of two condition runs (e.g. C0 vs C1) with Wilson CIs +
exact McNemar, and the list of tasks one condition fixed/broke.

Run: python3 scripts/compare_conditions.py \
       --a results/eval_<model>_c0_n40_k1.jsonl --a-name C0 \
       --b results/eval_<model>_c1_n40_k1.jsonl --b-name C1
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from da_verify.eval.stats import compare_paired


def _load(path: str) -> dict[int, bool]:
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    # per-task correctness; for k=1 this is the single outcome, for k>1 it's
    # "any sample correct" (pass@k). Comparisons here use matched k.
    return {r["id"]: (r["n_correct"] > 0) for r in rows}


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
        print(f"[warn] task sets differ; comparing the {len(ids)} shared ids only")

    cmp = compare_paired([A[i] for i in ids], [B[i] for i in ids])
    print("=" * 60)
    print(cmp.summary(args.a_name, args.b_name))
    print("-" * 60)
    print(f"{args.b_name} FIXED (A wrong → B right): {[i for i in ids if not A[i] and B[i]]}")
    print(f"{args.b_name} BROKE (A right → B wrong): {[i for i in ids if A[i] and not B[i]]}")


if __name__ == "__main__":
    main()
