"""Tools the agent can call, backed by the sandbox kernel.

Design note: `run_python` is the one real tool — everything an analysis needs can
be done through it. `list_columns` / `head` / `inspect_schema` are thin
convenience wrappers (they just run canned pandas in the same kernel). They earn
their place by saving the weak model from having to remember boilerplate to look
at the data, which is where small models otherwise waste steps or hallucinate
column names. The data is exposed as the variable CSV_PATH (set at kernel start).
"""

from __future__ import annotations

from ..sandbox import KernelSandbox

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Run Python in a stateful sandbox where pandas (pd), numpy (np) are "
                "imported and CSV_PATH points to the dataset. Variables persist across "
                "calls. Returns stdout / last-expression value / traceback. Use print() "
                "to see values."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python code to execute."}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_columns",
            "description": "List the dataset's column names (and dtypes).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "head",
            "description": "Show the first N rows of the dataset (default 5).",
            "parameters": {
                "type": "object",
                "properties": {"n": {"type": "integer", "description": "rows to show"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_schema",
            "description": "Show dataset shape, columns, dtypes, and basic null counts.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

_CANNED = {
    "list_columns": "import pandas as pd\n_df=pd.read_csv(CSV_PATH)\nprint({c:str(t) for c,t in _df.dtypes.items()})",
    "inspect_schema": (
        "import pandas as pd\n_df=pd.read_csv(CSV_PATH)\n"
        "print('shape:',_df.shape)\nprint('dtypes:')\nprint(_df.dtypes)\n"
        "print('nulls:')\nprint(_df.isnull().sum())"
    ),
}


def dispatch_tool(name: str, args: dict, sandbox: KernelSandbox) -> str:
    """Execute a tool call and return the observation string the agent sees."""
    if name == "run_python":
        code = (args or {}).get("code", "")
        if not code:
            return "error: run_python called with empty 'code'."
        return sandbox.execute(code).as_observation()
    if name == "head":
        n = int((args or {}).get("n", 5) or 5)
        return sandbox.execute(
            f"import pandas as pd\nprint(pd.read_csv(CSV_PATH).head({n}).to_string())"
        ).as_observation()
    if name in _CANNED:
        return sandbox.execute(_CANNED[name]).as_observation()
    return f"error: unknown tool {name!r}"
