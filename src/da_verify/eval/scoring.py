"""Single source of truth for scoring ONE agent response against a task.

Both runners (run_baseline, run_eval) call this, so the definition of
"correct / format-ok / candidate" lives in exactly one place and can't drift.
Uses VerifyResult.predicted directly (no second extract_answers pass).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..tasks.loader import Task
from ..tasks.verifier import CompareOptions, DEFAULT_OPTS, verify_response


@dataclass(frozen=True)
class SampleScore:
    correct: bool          # all required fields correct (the headline signal)
    format_ok: bool        # model emitted every required @name field
    candidate: bool        # model emitted ANY answer field at all
    n_correct_fields: int
    n_fields: int
    predicted: dict


def score_response(task: Task, response: str, opts: CompareOptions = DEFAULT_OPTS) -> SampleScore:
    vr = verify_response(task.id, response, [(g.name, g.value) for g in task.gold], opts)
    required = [g.name for g in task.gold]
    return SampleScore(
        correct=vr.all_correct,
        format_ok=all(r in vr.predicted for r in required),
        candidate=len(vr.predicted) > 0,
        n_correct_fields=vr.n_correct_fields,
        n_fields=len(required),
        predicted=vr.predicted,
    )
