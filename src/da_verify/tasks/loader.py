"""Load and join the InfiAgent-DABench (DAEval) public validation set.

WHAT THIS DOES (plain language):
  The benchmark ships two files that share an `id`:
    - da-dev-questions.jsonl : the question + constraints + required answer FORMAT
    - da-dev-labels.jsonl    : the gold answer(s) for that id
  This module reads both, joins them on `id`, and hands back immutable `Task`
  objects so the rest of the pipeline never touches raw JSON again.

WHY IT MATTERS:
  Everything downstream (sampling, the agent, the verifier, the stats) keys off
  a single clean representation. If the join is wrong, every number we ever
  report is wrong. So this is deliberately tiny and boring.

Source: https://github.com/InfiAgent/InfiAgent (examples/DA-Agent), ICML 2024.
Data license: CC BY-NC 4.0 (research/non-commercial use; attributed in NOTICE).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Repo-relative default location of the copied DAEval data.
_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "daeval"
QUESTIONS_PATH = _DATA_DIR / "da-dev-questions.jsonl"
LABELS_PATH = _DATA_DIR / "da-dev-labels.jsonl"
TABLES_DIR = _DATA_DIR / "da-dev-tables"


@dataclass(frozen=True)
class GoldAnswer:
    """One (name, value) pair. Multi-part questions carry several of these.

    `value` is kept as the RAW STRING from the label file (e.g. "34.65"),
    exactly as released. We never pre-parse it here — the verifier owns all
    type inference, so there is exactly one place where "what counts as equal"
    is decided.
    """

    name: str
    value: str


@dataclass(frozen=True)
class Task:
    """A single DAEval question joined with its gold label(s)."""

    id: int
    question: str
    concepts: tuple[str, ...]
    constraints: str
    answer_format: str
    file_name: str
    level: str  # "easy" | "medium" | "hard"
    gold: tuple[GoldAnswer, ...]

    @property
    def n_subanswers(self) -> int:
        """How many @name[value] fields this question expects (multi-part proxy)."""
        return len(self.gold)

    @property
    def table_path(self) -> Path:
        return TABLES_DIR / self.file_name


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"DAEval file missing: {path}\n"
            f"Expected the copied benchmark under {_DATA_DIR}. "
            f"See README for the fetch step."
        )
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON at {path}:{line_no}: {e}") from e
    return rows


def load_tasks(
    questions_path: Path = QUESTIONS_PATH,
    labels_path: Path = LABELS_PATH,
) -> list[Task]:
    """Read + join questions and labels into a list of Task, sorted by id.

    Fails loudly (KeyError/ValueError) if the two files disagree on ids — a
    silent mismatch here would corrupt every result, so we never paper over it.
    """
    questions = _read_jsonl(questions_path)
    labels_by_id = {row["id"]: row["common_answers"] for row in _read_jsonl(labels_path)}

    q_ids = {q["id"] for q in questions}
    l_ids = set(labels_by_id)
    if q_ids != l_ids:
        only_q, only_l = q_ids - l_ids, l_ids - q_ids
        raise ValueError(
            f"Question/label id mismatch — questions-only={sorted(only_q)[:5]}..., "
            f"labels-only={sorted(only_l)[:5]}..."
        )

    tasks: list[Task] = []
    for q in questions:
        raw_gold = labels_by_id[q["id"]]
        gold = tuple(GoldAnswer(name=str(name), value=str(value)) for name, value in raw_gold)
        tasks.append(
            Task(
                id=int(q["id"]),
                question=q["question"],
                concepts=tuple(q.get("concepts", [])),
                constraints=q.get("constraints", ""),
                answer_format=q.get("format", ""),
                file_name=q["file_name"],
                level=q["level"],
                gold=gold,
            )
        )
    return sorted(tasks, key=lambda t: t.id)


def tasks_by_id(tasks: list[Task]) -> dict[int, Task]:
    return {t.id: t for t in tasks}
