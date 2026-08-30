"""Inspect AI entry for the Inspect Evals Register.

The original study (`scripts/run_eval.py`) drives a stateful Jupyter kernel on
the host. Register security forbids `sandbox="local"` with model-produced
code, so this adapter runs the same C0–C3 policies through Inspect's Docker
sandbox and `python()` tool.

`python()` starts a fresh interpreter on every call. The agent must reload the
CSV in each cell. That is a deliberate protocol difference from the Jupyter
kernel; the grader and the C0–C3 policies are otherwise the same.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import Generate, Solver, TaskState, chain, solver, use_tools
from inspect_ai.tool import python

from da_verify.agent.react import (
    _USER,
    _VERIFIER_SYSTEM,
    _VERIFIER_USER,
    _VERIFY,
    _assemble_fields,
    _fields_agree,
    _majority_value,
)
from da_verify.eval.scoring import score_response
from da_verify.tasks.loader import GoldAnswer
from da_verify.tasks.loader import Task as DATask
from da_verify.tasks.loader import load_tasks, tasks_by_id
from da_verify.tasks.sampler import load_subset_ids

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBSET_PATH = REPO_ROOT / "data" / "subsets" / "headline_40.json"
COMPOSE_FILE = Path(__file__).parent / "sandbox" / "compose.yaml"

Condition = Literal["c0", "c1", "c2", "c3"]

_SYSTEM_INSPECT = """You are a careful data analyst. You answer questions about a dataset by writing and running Python.

Environment:
- A `python` tool that runs a FRESH interpreter on every call (no variables persist). pandas and numpy are installed.
- The dataset file is in the working directory. Load it with `import pandas as pd` then `pd.read_csv(<filename>)` at the start of every cell.
- Always print() values you need to see.

Answering rules:
- Follow the required answer FORMAT exactly. It looks like `@answer_name[value]`.
- Respect every constraint (rounding, which columns, etc.) precisely.
- When you have the answer, reply with the `@answer_name[value]` token(s) and nothing else. Do not call tools in that final message.
"""


def _da_task_from_metadata(metadata: dict[str, Any]) -> DATask:
    gold = tuple(
        GoldAnswer(name=str(name), value=str(value)) for name, value in metadata["gold"]
    )
    return DATask(
        id=int(metadata["task_id"]),
        question=metadata["question"],
        concepts=tuple(metadata.get("concepts") or ()),
        constraints=metadata.get("constraints") or "",
        answer_format=metadata.get("answer_format") or "",
        file_name=metadata["file_name"],
        level=metadata.get("level") or "medium",
        gold=gold,
    )


def score_completion(metadata: dict[str, Any], completion: str) -> dict[str, Any]:
    """Pure scoring helper so tests do not need a TaskState."""
    da_task = _da_task_from_metadata(metadata)
    sample = score_response(da_task, completion)
    return {
        "correct": sample.correct,
        "lenient_correct": sample.lenient_correct,
        "format_ok": sample.format_ok,
        "candidate": sample.candidate,
        "n_correct_fields": sample.n_correct_fields,
        "n_fields": sample.n_fields,
        "predicted": sample.predicted,
    }


def load_inspect_samples(n: int | None = 40) -> list[Sample]:
    """Headline subset as Inspect samples. Requires `bash scripts/fetch_data.sh`."""
    tasks = tasks_by_id(load_tasks())
    ids = load_subset_ids(SUBSET_PATH)
    if n is not None:
        ids = ids[:n]
    samples: list[Sample] = []
    for task_id in ids:
        da_task = tasks[task_id]
        csv_path = da_task.table_path
        if not csv_path.exists():
            raise FileNotFoundError(
                f"DAEval table missing: {csv_path}. Run: bash scripts/fetch_data.sh"
            )
        user = _USER.format(
            question=da_task.question,
            constraints=da_task.constraints,
            fmt=da_task.answer_format,
        )
        user = (
            f"{user}\n\nThe dataset filename is `{da_task.file_name}`. "
            "Reload it with pandas at the start of every python() call."
        )
        samples.append(
            Sample(
                id=da_task.id,
                input=user,
                target=", ".join(f"@{g.name}[{g.value}]" for g in da_task.gold),
                metadata={
                    "task_id": da_task.id,
                    "question": da_task.question,
                    "concepts": list(da_task.concepts),
                    "constraints": da_task.constraints,
                    "answer_format": da_task.answer_format,
                    "file_name": da_task.file_name,
                    "level": da_task.level,
                    "gold": [[g.name, g.value] for g in da_task.gold],
                },
                files={da_task.file_name: str(csv_path)},
            )
        )
    return samples


@solver
def da_verify_agent(condition: Condition = "c0") -> Solver:
    """C0–C3 policies using Inspect generate() + docker python()."""

    async def _run_loop(state: TaskState, generate: Generate) -> TaskState:
        return await generate(state, tool_calls="loop")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        metadata = state.metadata or {}
        da_task = _da_task_from_metadata(metadata)
        required = {g.name for g in da_task.gold}
        user = state.user_prompt.text

        async def independent_solve() -> str:
            state.messages = [
                ChatMessageSystem(content=_SYSTEM_INSPECT),
                ChatMessageUser(content=user),
            ]
            await _run_loop(state, generate)
            return state.output.completion or ""

        if condition == "c0":
            state.messages = [
                ChatMessageSystem(content=_SYSTEM_INSPECT),
                ChatMessageUser(content=user),
            ]
            return await _run_loop(state, generate)

        if condition == "c1":
            state.messages = [
                ChatMessageSystem(content=_SYSTEM_INSPECT),
                ChatMessageUser(content=user),
            ]
            await _run_loop(state, generate)
            if not (state.output.completion or "").strip():
                return state
            state.messages.append(ChatMessageUser(content=_VERIFY))
            return await _run_loop(state, generate)

        if condition == "c2":
            candidate = await independent_solve()
            if not candidate.strip():
                return state
            state.messages = [
                ChatMessageSystem(content=_VERIFIER_SYSTEM),
                ChatMessageUser(
                    content=_VERIFIER_USER.format(
                        question=da_task.question,
                        constraints=da_task.constraints,
                        fmt=da_task.answer_format,
                        candidate=candidate,
                    )
                ),
            ]
            await _run_loop(state, generate)
            vfinal = state.output.completion or ""
            from da_verify.tasks.verifier import extract_answers

            verifier_fields = set(extract_answers(vfinal)) if vfinal else set()
            if not required.issubset(verifier_fields):
                state.output.completion = candidate
            return state

        if condition == "c3":
            from da_verify.tasks.verifier import extract_answers

            first = await independent_solve()
            if not first.strip():
                return state
            answers = [extract_answers(first)]
            second = await independent_solve()
            answers.append(extract_answers(second))
            if _fields_agree(answers[0], answers[1], required):
                state.output.completion = first
                return state
            third = await independent_solve()
            answers.append(extract_answers(third))
            fields: dict[str, str] = {}
            for name in sorted(required):
                winner = _majority_value([a[name] for a in answers if name in a])
                if winner is None:
                    state.output.completion = first
                    return state
                fields[name] = winner
            state.output.completion = _assemble_fields(fields)
            return state

        raise ValueError(f"unknown condition {condition!r}")

    return solve


@scorer(metrics=[accuracy(), stderr()])
def da_verify_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        metadata = state.metadata or {}
        result = score_completion(metadata, state.output.completion or "")
        return Score(
            value=CORRECT if result["correct"] else INCORRECT,
            answer=state.output.completion,
            explanation=(
                f"correct={result['correct']} format_ok={result['format_ok']} "
                f"candidate={result['candidate']} "
                f"fields={result['n_correct_fields']}/{result['n_fields']}"
            ),
            metadata=result,
        )

    return score


@task
def da_verify(
    condition: Condition = "c0",
    n: int | None = 40,
) -> Task:
    """Verification-in-the-loop data-analysis eval on the DAEval headline subset.

    Args:
        condition: `c0` ReAct baseline, `c1` self-check, `c2` independent LLM
            verifier, `c3` programmatic agreement (no LLM judge).
        n: How many of the 40-task subset to run (default 40).
    """
    if condition not in ("c0", "c1", "c2", "c3"):
        raise ValueError("condition must be one of c0, c1, c2, c3")
    return Task(
        dataset=MemoryDataset(load_inspect_samples(n)),
        solver=chain(
            [
                use_tools(python(timeout=30)),
                da_verify_agent(condition),
            ]
        ),
        scorer=da_verify_scorer(),
        sandbox=("docker", str(COMPOSE_FILE)),
        message_limit=80,
    )
