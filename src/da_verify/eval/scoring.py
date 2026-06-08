"""Single source of truth for scoring ONE agent response against a task.

Both runners (run_baseline, run_eval) call this, so the definition of
"correct / format-ok / candidate" lives in exactly one place and can't drift.
Uses VerifyResult.predicted directly (no second extract_answers pass).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..tasks.loader import Task
from ..tasks.verifier import CompareOptions, DEFAULT_OPTS, compare_value, verify_response


@dataclass(frozen=True)
class SampleScore:
    correct: bool          # all required fields correct, right @name (the HEADLINE signal)
    lenient_correct: bool  # correct, OR (single-field) right VALUE under a wrong @name.
    #                        Reported SEPARATELY to quantify the format confound —
    #                        never folded into the headline `correct`.
    format_ok: bool        # model emitted every required @name field
    candidate: bool        # model emitted ANY answer field at all
    n_correct_fields: int
    n_fields: int
    predicted: dict


def score_response(task: Task, response: str, opts: CompareOptions = DEFAULT_OPTS) -> SampleScore:
    vr = verify_response(task.id, response, [(g.name, g.value) for g in task.gold], opts)
    required = [g.name for g in task.gold]

    # Format-forgiving credit, ONLY for the unambiguous single-field case: the
    # model gave exactly one answer whose VALUE is right but under the wrong
    # @name (e.g. id=132: @answer_count[20] vs gold @outlier_count[20]). We do
    # NOT do positional matching for multi-field answers — that would risk false
    # positives. This isolates "right analysis, wrong label" from "wrong analysis".
    lenient = vr.all_correct
    if not lenient and len(required) == 1 and len(vr.predicted) == 1:
        only_value = next(iter(vr.predicted.values()))
        lenient = compare_value(only_value, task.gold[0].value, opts)

    return SampleScore(
        correct=vr.all_correct,
        lenient_correct=lenient,
        format_ok=all(r in vr.predicted for r in required),
        candidate=len(vr.predicted) > 0,
        n_correct_fields=vr.n_correct_fields,
        n_fields=len(required),
        predicted=vr.predicted,
    )
