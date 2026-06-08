"""Tests for tool dispatch + the C0 ReAct loop, with mocked sandbox/LLM
(no real kernel or API). Covers the contract the agent silently relies on:
tool errors become observations, and the loop terminates correctly."""

from types import SimpleNamespace

from da_verify.agent.react import run_c0, run_c1
from da_verify.agent.tools import dispatch_tool
from da_verify.llm.client import LLMResponse
from da_verify.tasks.loader import GoldAnswer, Task


# ---- tool dispatch -------------------------------------------------------

class _FakeSandbox:
    def __init__(self):
        self.calls = []

    def execute(self, code):
        self.calls.append(code)
        return SimpleNamespace(as_observation=lambda: f"ran:{code[:20]}")


def test_dispatch_unknown_tool():
    assert "unknown tool" in dispatch_tool("bogus", {}, _FakeSandbox())


def test_dispatch_run_python_empty_code():
    assert "empty" in dispatch_tool("run_python", {"code": ""}, _FakeSandbox())


def test_dispatch_run_python_executes():
    sb = _FakeSandbox()
    out = dispatch_tool("run_python", {"code": "1+1"}, sb)
    assert out == "ran:1+1" and sb.calls == ["1+1"]


def test_dispatch_head_bad_arg_does_not_crash():
    # n=None / non-int must not raise — it clamps to a default.
    assert dispatch_tool("head", {"n": None}, _FakeSandbox()).startswith("ran:")


# ---- ReAct loop ----------------------------------------------------------

def _task():
    return Task(id=1, question="q", concepts=(), constraints="c", answer_format="@x[v]",
                file_name="t.csv", level="easy", gold=(GoldAnswer("x", "1"),))


class _ScriptedLLM:
    """Returns a pre-scripted list of LLMResponse, one per chat() call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.n = 0

    def chat(self, messages, tools=None, sample_id=0):
        r = self._responses[self.n]
        self.n += 1
        if isinstance(r, Exception):
            raise r
        return r


def _final(text):
    return LLMResponse(content=text, tool_calls=[], finish_reason="stop",
                       assistant_message={"role": "assistant", "content": text})


def _toolcall():
    tc = {"id": "c1", "name": "run_python", "arguments": {"code": "1"}, "arguments_raw": '{"code":"1"}'}
    am = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "run_python", "arguments": '{"code":"1"}'}}]}
    return LLMResponse(content="", tool_calls=[tc], finish_reason="tool_calls", assistant_message=am)


def test_immediate_final_answer():
    llm = _ScriptedLLM([_final("@x[1]")])
    tr = run_c0(_task(), llm, _FakeSandbox(), max_steps=5)
    assert tr.final_response == "@x[1]" and tr.steps == 1 and tr.tool_calls == 0


def test_toolcall_then_answer():
    llm = _ScriptedLLM([_toolcall(), _final("@x[1]")])
    tr = run_c0(_task(), llm, _FakeSandbox(), max_steps=5)
    assert tr.final_response == "@x[1]" and tr.tool_calls == 1 and tr.steps == 2


def test_api_error_is_captured_not_raised():
    llm = _ScriptedLLM([RuntimeError("boom")])
    tr = run_c0(_task(), llm, _FakeSandbox(), max_steps=5)
    assert tr.error and "boom" in tr.error and tr.final_response == ""


def test_hit_max_steps_then_final_fallback():
    # always tool-calls -> exhausts steps -> one last non-tool answer
    llm = _ScriptedLLM([_toolcall(), _toolcall(), _final("@x[1]")])
    tr = run_c0(_task(), llm, _FakeSandbox(), max_steps=2)
    assert tr.hit_max_steps and tr.final_response == "@x[1]"


# ---- C1 self-verification ------------------------------------------------

def test_c1_runs_verification_round_and_can_revise():
    # phase 1: tool-call then candidate @x[1]; phase 2 (after self-check): revise to @x[2]
    llm = _ScriptedLLM([_toolcall(), _final("@x[1]"), _final("@x[2]")])
    tr = run_c1(_task(), llm, _FakeSandbox(), max_steps=5)
    assert tr.condition == "c1"
    assert tr.final_response == "@x[2]"   # the post-verification answer is used


def test_c1_skips_verification_when_no_candidate():
    # phase 1 gives no answer -> nothing to verify -> verification round NOT entered
    llm = _ScriptedLLM([_final("")])
    tr = run_c1(_task(), llm, _FakeSandbox(), max_steps=5)
    assert tr.condition == "c1" and tr.final_response == ""
    assert llm.n == 1
