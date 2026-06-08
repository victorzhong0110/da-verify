"""Tests for stratified sampling — subset reproducibility is a research claim,
so determinism (same seed -> same ids) must be pinned, and silent under-fill caught."""

import pytest

from da_verify.tasks.loader import GoldAnswer, Task
from da_verify.tasks.sampler import stratified_sample


def _mk(i, level, n_gold=1, categorical=False):
    val = "no" if categorical else "1.0"
    gold = tuple(GoldAnswer(name=f"f{j}", value=val) for j in range(n_gold))
    return Task(id=i, question="q", concepts=("A",), constraints="", answer_format="",
                file_name="t.csv", level=level, gold=gold)


def _population():
    tasks = []
    i = 0
    for level in ("easy", "medium", "hard"):
        for _ in range(20):
            tasks.append(_mk(i, level, n_gold=(1 if i % 2 else 2)))
            i += 1
    return tasks


def test_deterministic_same_seed():
    pop = _population()
    a = [t.id for t in stratified_sample(pop, n=12, seed=42)]
    b = [t.id for t in stratified_sample(pop, n=12, seed=42)]
    assert a == b


def test_different_seed_can_differ():
    pop = _population()
    a = [t.id for t in stratified_sample(pop, n=12, seed=1)]
    b = [t.id for t in stratified_sample(pop, n=12, seed=2)]
    assert a != b  # not guaranteed in theory, but overwhelmingly likely here


def test_spans_all_levels():
    pop = _population()
    chosen = stratified_sample(pop, n=12, seed=7)
    levels = {t.level for t in chosen}
    assert levels == {"easy", "medium", "hard"}


def test_warns_when_pool_exhausted():
    pop = _population()  # 60 tasks
    with pytest.warns(UserWarning, match="only"):
        chosen = stratified_sample(pop, n=100, seed=0)
    assert len(chosen) <= 60
