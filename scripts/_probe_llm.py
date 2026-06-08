"""One-off probe: validate key, find a working (base_url, model), confirm
tool-calling + @name[value] format adherence. Prints status only (never the key)."""

import os
from pathlib import Path

for line in (Path(__file__).resolve().parents[1] / ".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from openai import OpenAI

KEY = os.environ["MINIMAX_API_KEY"]
BASE_URLS = ["https://api.minimax.io/v1", "https://api.minimaxi.com/v1"]
MODELS = ["MiniMax-M2.7", "MiniMax-M2.5", "MiniMax-M2"]

tools = [{
    "type": "function",
    "function": {
        "name": "run_python",
        "description": "Execute python code and return stdout.",
        "parameters": {"type": "object",
                       "properties": {"code": {"type": "string"}},
                       "required": ["code"]},
    },
}]


def probe(base, model):
    c = OpenAI(api_key=KEY, base_url=base, timeout=60)
    r = c.chat.completions.create(
        model=model,
        messages=[{"role": "user",
                   "content": "Use the run_python tool to compute the mean of [2,4,6]."}],
        tools=tools, temperature=0, max_tokens=300,
    )
    msg = r.choices[0].message
    tc = getattr(msg, "tool_calls", None)
    return r.choices[0].finish_reason, bool(tc), (tc[0].function.name if tc else None)


for base in BASE_URLS:
    for model in MODELS:
        try:
            finish, has_tc, tname = probe(base, model)
            print(f"OK   base={base:32} model={model:14} tool_calls={'yes' if has_tc else 'no '} "
                  f"(call={tname}) finish={finish}")
        except Exception as e:
            print(f"FAIL base={base:32} model={model:14} {type(e).__name__}: {str(e)[:80]}")
