"""Build the headline ~40-question subset and save its ids. Deterministic.

Run:  python3 scripts/make_subset.py
Out:  data/subsets/headline_40.json  (+ a printed stratum table)
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (puts src/ on path)

from pathlib import Path
from collections import Counter

from da_verify.tasks.loader import load_tasks
from da_verify.tasks.sampler import (
    complexity_bucket,
    has_noisy_categorical,
    save_subset,
    stratified_sample,
)
from da_verify.tasks.verifier import infer_type

ROOT = Path(__file__).resolve().parents[1]
SUBSET_PATH = ROOT / "data" / "subsets" / "headline_40.json"

N = 40
SEED = 20260607


def _table(title, counter):
    print(f"\n{title}")
    for k, v in sorted(counter.items(), key=lambda x: str(x[0])):
        print(f"   {str(k):<22} {v}")


def main() -> None:
    tasks = load_tasks()
    subset = stratified_sample(tasks, n=N, seed=SEED, prefer_verifiable=True)

    print(f"Full set: {len(tasks)} tasks   ->   subset: {len(subset)} tasks")
    _table("Full-set level:", Counter(t.level for t in tasks))
    _table("Subset level:", Counter(t.level for t in subset))
    _table("Subset complexity:", Counter(complexity_bucket(t) for t in subset))
    _table(
        "Subset answer-type (by field):",
        Counter(infer_type(g.value) for t in subset for g in t.gold),
    )
    print(f"\nSubset tasks with any noisy categorical: "
          f"{sum(has_noisy_categorical(t) for t in subset)} / {len(subset)}")

    save_subset(
        subset,
        SUBSET_PATH,
        meta={
            "benchmark": "InfiAgent-DABench / DAEval (public validation)",
            "n": len(subset),
            "seed": SEED,
            "strata": "level x complexity (single/multi)",
            "scoping": "verifiability-first: prefer numeric/clean-list over noisy categorical",
        },
    )
    print(f"\nSaved -> {SUBSET_PATH.relative_to(ROOT)}")
    print("ids:", [t.id for t in subset])


if __name__ == "__main__":
    main()
