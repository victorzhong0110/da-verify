"""C0 baseline agent: a hand-written ReAct loop (Thought -> Action -> Observation).

C0 is the *no-verification* baseline by design — the system prompt does NOT ask
the model to double-check itself. C1 (self-check) and C2 (multi-step verify) are
deliberately built later as deltas on top of this exact loop, so the only thing
that changes between conditions is the verification policy. Keeping C0 honest
(zero self-check nudge) is what makes the eventual C0-vs-C2 comparison clean.

We hand-write the loop (rather than using a framework) so every step — the stop
condition, the message bookkeeping, the tool dispatch — is visible and defensible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.client import LLMClient
from ..sandbox import KernelSandbox
from ..tasks.loader import Task
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


@dataclass
class RunTrace:
    task_id: int
    final_response: str
    steps: int
    tool_calls: int
    transcript: list[dict] = field(default_factory=list)
    hit_max_steps: bool = False
    error: str | None = None


def run_c0(task: Task, llm: LLMClient, sandbox: KernelSandbox, max_steps: int = 8,
           sample_id: int = 0) -> RunTrace:
    """Run the C0 loop on one task. `task` is a da_verify.tasks.Task.

    sample_id labels which of the k pass@k repeats this is, so each repeat is a
    distinct (cache-separated) draw.
    """
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _USER.format(
            question=task.question, constraints=task.constraints, fmt=task.answer_format)},
    ]
    n_tool_calls = 0
    for step in range(1, max_steps + 1):
        try:
            resp = llm.chat(messages, tools=TOOL_SCHEMAS, sample_id=sample_id)
        except Exception as e:  # API/network failure — surface, don't pretend success
            return RunTrace(task.id, "", step, n_tool_calls, messages, error=f"{type(e).__name__}: {e}")

        messages.append(resp.assistant_message)

        if not resp.tool_calls:
            # No tool call => the model is giving its final answer.
            return RunTrace(task.id, resp.content or "", step, n_tool_calls, messages)

        for tc in resp.tool_calls:
            n_tool_calls += 1
            args = tc["arguments"] if tc["arguments"] is not None else {}
            try:
                obs = dispatch_tool(tc["name"], args, sandbox)
            except Exception as e:
                # A tool failure must become an observation the model can react to,
                # never an exception that crashes the whole eval run.
                obs = f"error: tool '{tc['name']}' failed: {type(e).__name__}: {e}"
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": obs})

    # Ran out of steps. Make one last non-tool request for a final answer.
    messages.append({"role": "user",
                     "content": "Stop using tools. Give your final answer now in the exact @name[value] format."})
    try:
        final = llm.chat(messages, tools=None, sample_id=sample_id)
        return RunTrace(task.id, final.content or "", max_steps, n_tool_calls, messages, hit_max_steps=True)
    except Exception as e:
        return RunTrace(task.id, "", max_steps, n_tool_calls, messages, hit_max_steps=True,
                        error=f"{type(e).__name__}: {e}")
