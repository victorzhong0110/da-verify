"""The agent loop and the verification conditions.

The controlled study's whole validity rests on ONE rule: across conditions, the
ONLY thing that changes is the verification policy. Everything else — model,
tools, base prompt, task, decoding — is held fixed. So the conditions are built
as deltas on a single shared loop (`_react_loop`):

  C0  no verification  — run to a final answer, no self-check nudge (baseline).
  C1  self-verification — C0, then ONE reflexion round: the same agent re-checks
                          its own code+answer and may revise once.
  (C2 multi-step / independent verify lands in W5.)

We hand-write the loop so every step (stop condition, message bookkeeping, tool
dispatch, error handling) is visible and defensible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.client import LLMClient
from ..sandbox import KernelSandbox
from ..tasks.loader import Task
from ..tasks.verifier import extract_answers
from .tools import TOOL_SCHEMAS, dispatch_tool

_SYSTEM = """You are a careful data analyst. You answer questions about a dataset \
by writing and running Python.

Environment:
- A stateful Python sandbox: pandas is imported as `pd`, numpy as `np`.
- The dataset path is in the variable `CSV_PATH`. Load it with `pd.read_csv(CSV_PATH)`.
- Use the tools to inspect the data and run code. Always print() values you need to see.

Answering rules:
- Follow the required answer FORMAT exactly. It looks like `@answer_name[value]`.
- Respect every constraint (rounding, which columns, etc.) precisely.
- When you have the answer, reply with the `@answer_name[value]` token(s) and nothing else.
"""

_USER = """Question:
{question}

Constraints:
{constraints}

Required answer format:
{fmt}

Work it out using the tools, then give the final answer in the exact format above."""

# C1 self-verification. Deliberately NOT told the answer is wrong (that would
# bias toward changing it). It asks for an independent double-check.
_VERIFY = """Before you finalize, double-check your own work:
- Did you use the correct column(s) the question asks about?
- Did you handle missing values / data types correctly, with no data leakage?
- Did you follow every constraint (rounding, filters, method) exactly?
- Re-derive or sanity-check the key number(s) if useful (you may run code again).
- Does the answer use the EXACT required @answer_name[value] format and field name?

If you find a mistake, correct it. Otherwise restate the same final answer in the exact format."""

# C2 external verification: a FRESH, skeptical analyst that independently recomputes
# (in its own sandbox, from raw data) rather than introspecting. The W4 null showed
# self-critique adds no signal; an independent re-derivation by executing code is the
# external signal. It is told the candidate ONLY so it can reconcile — and told not to
# trust it.
_VERIFIER_SYSTEM = """You are an INDEPENDENT verifier — a skeptical second analyst.
You are given a question about a dataset and a CANDIDATE answer produced by someone
else. Do NOT assume the candidate is correct.

Your job: recompute the answer yourself, from scratch, from the raw data \
(load it fresh with pd.read_csv(CSV_PATH)). Prefer a DIFFERENT method than the most \
obvious one, and run sanity checks (row counts, null handling, value ranges). Then \
decide the correct value.

Environment: stateful sandbox, pandas as pd, numpy as np, data at CSV_PATH; use the \
tools to run code. Finish with the CORRECT answer in the exact required \
@answer_name[value] format — your INDEPENDENTLY verified value, whether or not it \
matches the candidate."""

_VERIFIER_USER = """Question:
{question}

Constraints:
{constraints}

Required answer format:
{fmt}

CANDIDATE answer (verify independently — do NOT assume it is correct):
{candidate}

Recompute from the raw data yourself, cross-check, then give the correct answer in the exact format."""


@dataclass
class RunTrace:
    task_id: int
    final_response: str
    steps: int
    tool_calls: int
    transcript: list[dict] = field(default_factory=list)
    hit_max_steps: bool = False
    error: str | None = None
    condition: str = "c0"


def _react_loop(messages: list[dict], llm: LLMClient, sandbox: KernelSandbox,
                max_steps: int, sample_id: int, n_tool_calls: int = 0):
    """Run Thought->Action->Observation on `messages` (mutated in place) until the
    model answers without a tool call, or max_steps is hit. Returns
    (final_response, steps, n_tool_calls, hit_max_steps, error)."""
    for step in range(1, max_steps + 1):
        try:
            resp = llm.chat(messages, tools=TOOL_SCHEMAS, sample_id=sample_id)
        except Exception as e:  # API/network failure — surface, don't fake success
            return "", step, n_tool_calls, False, f"{type(e).__name__}: {e}"

        messages.append(resp.assistant_message)
        if not resp.tool_calls:
            return resp.content or "", step, n_tool_calls, False, None

        for tc in resp.tool_calls:
            n_tool_calls += 1
            args = tc["arguments"] if tc["arguments"] is not None else {}
            try:
                obs = dispatch_tool(tc["name"], args, sandbox)
            except Exception as e:
                # tool failure becomes an observation, never crashes the run
                obs = f"error: tool '{tc['name']}' failed: {type(e).__name__}: {e}"
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": obs})

    # ran out of steps -> one last non-tool request for a final answer
    messages.append({"role": "user",
                     "content": "Stop using tools. Give your final answer now in the exact @name[value] format."})
    try:
        final = llm.chat(messages, tools=None, sample_id=sample_id)
        return final.content or "", max_steps, n_tool_calls, True, None
    except Exception as e:
        return "", max_steps, n_tool_calls, True, f"{type(e).__name__}: {e}"


def _init_messages(task: Task) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _USER.format(
            question=task.question, constraints=task.constraints, fmt=task.answer_format)},
    ]


def run_c0(task: Task, llm: LLMClient, sandbox: KernelSandbox, max_steps: int = 8,
           sample_id: int = 0) -> RunTrace:
    """C0 baseline: run to a final answer, no self-check."""
    messages = _init_messages(task)
    final, steps, ntc, hit_max, err = _react_loop(messages, llm, sandbox, max_steps, sample_id)
    return RunTrace(task.id, final, steps, ntc, messages, hit_max, err, condition="c0")


def run_c1(task: Task, llm: LLMClient, sandbox: KernelSandbox, max_steps: int = 8,
           sample_id: int = 0, verify_steps: int = 4) -> RunTrace:
    """C1: C0 followed by ONE self-verification round.

    Phase 1 is byte-identical to C0 (same messages) -> replays from cache when C0
    was already run. Phase 2 adds the self-check prompt and a short continuation.
    """
    messages = _init_messages(task)
    final, steps, ntc, hit_max, err = _react_loop(messages, llm, sandbox, max_steps, sample_id)
    if err or not final.strip():
        # nothing to verify (errored or produced no candidate) -> C1 == C0 here
        return RunTrace(task.id, final, steps, ntc, messages, hit_max, err, condition="c1")

    messages.append({"role": "user", "content": _VERIFY})
    final2, steps2, ntc2, hit_max2, err2 = _react_loop(
        messages, llm, sandbox, verify_steps, sample_id, n_tool_calls=ntc)
    return RunTrace(
        task.id,
        final2 or final,  # fall back to the pre-verification answer if phase 2 errored
        steps + steps2,
        ntc2,
        messages,
        hit_max or hit_max2,
        err2,
        condition="c1",
    )


def run_c2(task: Task, llm: LLMClient, sandbox: KernelSandbox, max_steps: int = 8,
           sample_id: int = 0, verify_steps: int = 6, verifier_llm: LLMClient | None = None) -> RunTrace:
    """C2: C0, then an INDEPENDENT verifier that recomputes from scratch in its
    own fresh sandbox (external signal = re-execution), and reconciles.

    Distinct from C1: not the same agent introspecting, but a skeptical second
    pass that runs its own code. Its independently-derived answer is the output.
    `verifier_llm` lets the verifier be a DIFFERENT (e.g. stronger) model than the
    solver — the multi-model verification lever.
    """
    verifier_llm = verifier_llm or llm
    messages = _init_messages(task)
    final, steps, ntc, hit_max, err = _react_loop(messages, llm, sandbox, max_steps, sample_id)
    if err or not final.strip():
        return RunTrace(task.id, final, steps, ntc, messages, hit_max, err, condition="c2")

    vmsgs = [
        {"role": "system", "content": _VERIFIER_SYSTEM},
        {"role": "user", "content": _VERIFIER_USER.format(
            question=task.question, constraints=task.constraints,
            fmt=task.answer_format, candidate=final)},
    ]
    # fresh sandbox so the verifier can't be contaminated by the solver's state
    with KernelSandbox(data_csv=task.table_path) as vsb:
        vfinal, vsteps, vntc, vhit, verr = _react_loop(
            vmsgs, verifier_llm, vsb, verify_steps, sample_id)

    # Reconciliation: adopt the verifier's answer ONLY if it is COMPLETE — it must
    # cover EVERY required field. A verifier that re-derives only some fields
    # (common on multi-part questions, especially across models) must not override
    # a complete candidate with a partial answer — that destroys correct fields.
    # (Subsumes the earlier rule: a non-answer covers nothing, so it can't override.)
    required = {g.name for g in task.gold}
    verifier_fields = set(extract_answers(vfinal)) if vfinal else set()
    use_verifier = required.issubset(verifier_fields)
    final_answer = vfinal if use_verifier else final

    return RunTrace(
        task.id,
        final_answer,
        steps + vsteps,
        ntc + vntc,
        messages + vmsgs,
        hit_max or vhit,
        verr,
        condition="c2",
    )
