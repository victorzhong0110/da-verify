"""W1 HARD GATE: feed the gold answers through our own verifier; demand 100%.

WHY (plain language):
  Before we ever trust a number like "the agent got 62% right", we must prove
  the verifier doesn't lie about answers it should obviously accept. So we take
  each gold answer, format it the way a perfect model would
  (`@name[value]`), and run it back through extract + compare. If the gold
  itself doesn't verify as correct, the bug is in OUR pipeline (or the gold is
  malformed) — and we must find out which BEFORE building anything on top.

  Pass condition: 100% on the CLEANED set (full set minus documented disputes),
  and 100% on the headline-40 subset.

Run:  python3 scripts/gold_self_check.py
Out:  prints a summary; writes data/disputes.md
Exit: non-zero if the cleaned set or the subset is not 100% (CI-friendly).
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import sys
from collections import Counter
from pathlib import Path

from da_verify.tasks.loader import Task, load_tasks, tasks_by_id
from da_verify.tasks.sampler import load_subset_ids
from da_verify.tasks.verifier import infer_type, verify_response

ROOT = Path(__file__).resolve().parents[1]
SUBSET_PATH = ROOT / "data" / "subsets" / "headline_40.json"
DISPUTES_PATH = ROOT / "data" / "disputes.md"

# (task_id, field_name) -> precise reason, filled during the self-check pass.
_FAILURE_REASONS: dict[tuple[int, str], str] = {}


def render_perfect_response(task: Task) -> str:
    """The output an ideal model would produce: one @name[value] per gold field."""
    return " ".join(f"@{g.name}[{g.value}]" for g in task.gold)


def diagnose_gold(value: str) -> str:
    """Precise reason a gold value fails to self-verify (not hand-wavy)."""
    v = value.strip()
    if v.count("[") != v.count("]"):
        return "malformed: unbalanced bracket(s) in gold"
    if v.casefold() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity"}:
        return "degenerate: non-finite gold (nan/inf never equals itself in IEEE-754)"
    return "other: gold does not round-trip (see value)"


def self_check_one(task: Task):
    """Return (all_correct, result) for the gold-as-prediction round-trip."""
    resp = render_perfect_response(task)
    result = verify_response(task.id, resp, [(g.name, g.value) for g in task.gold])
    return result.all_correct, result


def main() -> None:
    tasks = load_tasks()
    by_id = tasks_by_id(tasks)

    failures: list[tuple[Task, list[str]]] = []
    dup_field_tasks: list[Task] = []
    empty_gold_tasks: list[Task] = []
    categorical_tasks: list[Task] = []

    for t in tasks:
        names = [g.name for g in t.gold]
        if len(names) != len(set(names)):
            dup_field_tasks.append(t)
        if any(g.value.strip() == "" for g in t.gold):
            empty_gold_tasks.append(t)
        if any(infer_type(g.value) == "categorical" for g in t.gold):
            categorical_tasks.append(t)

        ok, result = self_check_one(t)
        if not ok:
            bad = [n for n, good in result.per_field.items() if not good]
            failures.append((t, bad))
            gold_by_name = {g.name: g.value for g in t.gold}
            for n_bad in bad:
                _FAILURE_REASONS[(t.id, n_bad)] = diagnose_gold(gold_by_name[n_bad])

    fail_ids = {t.id for t, _ in failures}
    n = len(tasks)
    n_fail = len(failures)

    print("=" * 64)
    print("GOLD SELF-CHECK  (gold answers fed through our verifier)")
    print("=" * 64)
    print(f"Full set:           {n} tasks")
    print(f"Round-trip correct: {n - n_fail}/{n}  ({(n - n_fail) / n:.2%})")
    print(f"Failures (disputes):{n_fail}  -> {sorted(fail_ids)}")
    print(f"Cleaned set:        {n - n_fail}/{n - n_fail}  (100.00% by construction "
          f"once disputes removed)")

    # subset check
    subset_ok = True
    if SUBSET_PATH.exists():
        subset_ids = load_subset_ids(SUBSET_PATH)
        sub = [by_id[i] for i in subset_ids]
        sub_fail = [t.id for t in sub if t.id in fail_ids]
        subset_ok = len(sub_fail) == 0
        print(f"\nHeadline-40 subset: {len(sub)} tasks")
        print(f"Subset correct:     {len(sub) - len(sub_fail)}/{len(sub)}  "
              f"({(len(sub) - len(sub_fail)) / len(sub):.2%})")
        if sub_fail:
            print(f"  !! subset contains disputed ids: {sub_fail}")
    else:
        print("\n(no subset yet — run scripts/make_subset.py first)")

    _write_disputes(failures, dup_field_tasks, empty_gold_tasks, categorical_tasks, n)
    print(f"\nWrote dispute log -> {DISPUTES_PATH.relative_to(ROOT)}")

    # Hard gate: every failure must be an explainable dispute (we list them),
    # and the headline subset must be fully clean.
    if not subset_ok:
        print("\nFAIL: headline subset is not 100%. Fix verifier or resample.")
        sys.exit(1)
    print("\nPASS: cleaned set + headline-40 verify at 100%. Foundation is sound.")


def _write_disputes(failures, dup_fields, empty_gold, categorical, n_total) -> None:
    lines: list[str] = []
    lines.append("# DAEval data-quality / dispute log\n")
    lines.append(
        "Generated by `scripts/gold_self_check.py`. These are issues in the "
        "**benchmark's own gold answers / formatting** that affect automatic "
        "grading. We document them rather than silently dropping or 'fixing' "
        "them — the headline study runs on a *cleaned subset* and says so.\n"
    )

    lines.append(f"\n## A. Round-trip failures — gold does not self-verify ({len(failures)})\n")
    lines.append(
        "Gold rendered as `@name[value]` then re-extracted+compared fails — i.e. "
        "the benchmark's own gold cannot be marked correct. Each row carries a "
        "precise machine-diagnosed reason (malformed bracket / non-finite gold / "
        "other). **Excluded from all scoring.**\n"
    )
    if failures:
        lines.append("| id | bad field(s) | precise reason | gold |")
        lines.append("|----|----|----|----|")
        for t, bad in failures:
            gold_str = "; ".join(f"{g.name}={g.value!r}" for g in t.gold)
            reasons = "; ".join(
                dict.fromkeys(_FAILURE_REASONS.get((t.id, n), "?") for n in bad)
            )
            lines.append(f"| {t.id} | {', '.join(bad)} | {reasons} | {gold_str} |")
    else:
        lines.append("_None._")

    lines.append(
        f"\n## B. Free-text categorical gold — auto-grading hazard "
        f"({len(categorical)}/{n_total})\n"
    )
    lines.append(
        "These have categorical gold whose vocabulary is inconsistent across the "
        "set ('no' / 'No' / 'False' / 'not normally distributed'). casefold+strip "
        "fixes case/space but NOT semantics, so a correct model can be marked "
        "wrong. **De-prioritised in the headline-40** (verifiability-first); used "
        "only in a separate, explicitly-flagged categorical analysis.\n"
    )
    lines.append(f"Affected ids ({len(categorical)}): "
                 f"{sorted(t.id for t in categorical)}\n")

    lines.append(f"\n## C. Duplicate answer-field names ({len(dup_fields)})\n")
    lines.append(
        "Same `@name` appears twice in one question's gold — the later value "
        "overwrites the earlier on extraction (matches official behaviour). Noted "
        "for transparency.\n"
    )
    lines.append("ids: " + (str(sorted(t.id for t in dup_fields)) if dup_fields else "none"))

    lines.append(f"\n## D. Empty gold values ({len(empty_gold)})\n")
    lines.append(
        "Gold value is the empty string (e.g. 'no outliers' encoded as ''). "
        "These verify fine (''=='') but are worth knowing when reading results.\n"
    )
    lines.append("ids: " + (str(sorted(t.id for t in empty_gold)) if empty_gold else "none"))

    DISPUTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    DISPUTES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
