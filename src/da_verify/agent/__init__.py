from .react import RunTrace, run_c0, run_c1, run_c2, run_c3
from .tools import TOOL_SCHEMAS, dispatch_tool

__all__ = ["TOOL_SCHEMAS", "dispatch_tool", "run_c0", "run_c1", "run_c2", "run_c3", "RunTrace"]

# Verification conditions, keyed for the runner:
#   c0 no verification | c1 self-verification | c2 independent external (LLM) verification
#   c3 programmatic verification (self-consistency agreement gate, no LLM judge)
CONDITIONS = {"c0": run_c0, "c1": run_c1, "c2": run_c2, "c3": run_c3}
