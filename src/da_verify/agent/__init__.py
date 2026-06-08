from .react import RunTrace, run_c0, run_c1, run_c2
from .tools import TOOL_SCHEMAS, dispatch_tool

__all__ = ["TOOL_SCHEMAS", "dispatch_tool", "run_c0", "run_c1", "run_c2", "RunTrace"]

# Verification conditions, keyed for the runner:
#   c0 no verification | c1 self-verification | c2 independent external verification
CONDITIONS = {"c0": run_c0, "c1": run_c1, "c2": run_c2}
