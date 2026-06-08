"""Tests for the LLM client's cache — pass@k reproducibility depends on the
cache key being correct, especially the subtle sample_id=0 backward-compat rule."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from da_verify.llm.client import LLMClient


def _client(tmp_path):
    return LLMClient(model="m", api_key="x", base_url="http://localhost", cache_dir=tmp_path)


def _fake_resp(content="@x[1]", tool_args=None):
    tcs = None
    if tool_args is not None:
        tcs = [SimpleNamespace(id="c1", function=SimpleNamespace(name="run_python", arguments=tool_args))]
    msg = SimpleNamespace(content=content, tool_calls=tcs)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))


def test_sample_id_zero_is_backward_compatible(tmp_path):
    c = _client(tmp_path)
    msgs = [{"role": "user", "content": "hi"}]
    # sample_id=0 must hash IDENTICALLY to the W2-era key that had no sample_id field
    expected = hashlib.sha256(
        json.dumps({"model": "m", "messages": msgs, "tools": None,
                    "temperature": 0.0, "max_tokens": 2048},
                   sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    assert c._cache_key(msgs, None, 0) == expected


def test_sample_id_distinguishes_repeats(tmp_path):
    c = _client(tmp_path)
    msgs = [{"role": "user", "content": "hi"}]
    assert c._cache_key(msgs, None, 0) != c._cache_key(msgs, None, 1)
    assert c._cache_key(msgs, None, 1) != c._cache_key(msgs, None, 2)


def test_record_then_replay(tmp_path, monkeypatch):
    c = _client(tmp_path)
    calls = {"n": 0}

    def fake_create(**kw):
        calls["n"] += 1
        return _fake_resp()

    monkeypatch.setattr(c._client.chat.completions, "create", fake_create)
    r1 = c.chat([{"role": "user", "content": "hi"}])
    assert r1.cached is False and calls["n"] == 1 and r1.content == "@x[1]"
    r2 = c.chat([{"role": "user", "content": "hi"}])  # same request
    assert r2.cached is True and calls["n"] == 1       # served from disk, no 2nd API call


def test_malformed_tool_args_surface_as_none(tmp_path):
    # Bad JSON in tool arguments must not crash; parsed -> None, raw kept.
    resp = _fake_resp(content=None, tool_args="{not json}")
    norm = LLMClient._normalise(resp)
    assert norm.tool_calls[0]["arguments"] is None
    assert norm.tool_calls[0]["arguments_raw"] == "{not json}"


def test_empty_choices_raises():
    with pytest.raises(ValueError, match="no choices"):
        LLMClient._normalise(SimpleNamespace(choices=[]))
