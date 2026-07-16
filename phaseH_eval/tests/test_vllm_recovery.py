"""Unit tests for VllmDirectModel one-shot recovery mode (no server needed)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import vllm_direct_model as vdm  # noqa: E402
from vllm_direct_model import (  # noqa: E402
    RECOVERY_INSTRUCTION,
    RECOVERY_MAX_TOKENS,
    VllmDirectModel,
    build_recovery_kwargs,
    response_needs_recovery,
)


def _resp(finish_reason, *, tool_calls=None, content=""):
    msg = SimpleNamespace(tool_calls=tool_calls, content=content)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason, message=msg)])


class FakeClient:
    """Records create() kwargs and returns a scripted queue of responses."""
    def __init__(self, queue):
        self.queue = list(queue)
        self.calls = []

    @property
    def chat(self):
        return SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.queue.pop(0)


def _bare_model(client):
    m = VllmDirectModel.__new__(VllmDirectModel)
    m._client = client
    m._call_kwargs = {"max_tokens": 4096, "temperature": 0.7}
    m._recovery_armed = False
    m.config = SimpleNamespace(model_name="openai/gemma4-rawbase")
    return m


# --- helper coverage ---

def test_response_needs_recovery_length_no_tool():
    assert response_needs_recovery(_resp("length", tool_calls=None, content="just prose, no command"))


def test_response_needs_recovery_false_when_tool_present():
    assert not response_needs_recovery(_resp("length", tool_calls=[object()]))


def test_response_needs_recovery_false_when_not_length():
    assert not response_needs_recovery(_resp("stop", tool_calls=None, content="prose"))


def test_build_recovery_kwargs_shape():
    call_kwargs, create_kwargs = build_recovery_kwargs({"max_tokens": 4096, "temperature": 0.7, "extra_body": {"x": 1}})
    assert call_kwargs["max_tokens"] == RECOVERY_MAX_TOKENS == 1024
    assert call_kwargs["tool_choice"] == {"type": "function", "function": {"name": "bash"}}
    assert "extra_body" not in call_kwargs
    assert create_kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


# --- the three required state-machine tests ---

def test_1_length_no_tool_arms_recovery():
    client = FakeClient([_resp("length", tool_calls=None, content="I need to think more about this...")])
    m = _bare_model(client)
    m._query([{"role": "user", "content": "fix the bug"}])
    assert m._recovery_armed is True


def test_2_recovery_request_forces_bash_1024_thinking_off():
    # first response arms; second is the recovery retry whose kwargs we inspect
    client = FakeClient([
        _resp("length", tool_calls=None, content="rambling, no command"),
        _resp("stop", tool_calls=[object()]),
    ])
    m = _bare_model(client)
    m._query([{"role": "user", "content": "fix"}])            # arms
    assert m._recovery_armed is True
    m._query([{"role": "user", "content": "fix"}])            # recovery retry
    recovery_call = client.calls[1]
    assert recovery_call["max_tokens"] == 1024
    assert recovery_call["tool_choice"] == {"type": "function", "function": {"name": "bash"}}
    assert recovery_call["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert recovery_call["messages"][-1] == {"role": "user", "content": RECOVERY_INSTRUCTION}
    assert m._recovery_armed is False  # one-shot consumed


def test_3_normal_tool_call_never_arms():
    client = FakeClient([_resp("stop", tool_calls=[object()])])
    m = _bare_model(client)
    m._query([{"role": "user", "content": "fix"}])
    assert m._recovery_armed is False
    # and a normal request carries no forced tool_choice / recovery extra_body
    assert "tool_choice" not in client.calls[0]


def test_recovery_failure_does_not_rearm():
    # recovery retry that STILL length+no-tool must fall through, not re-arm
    client = FakeClient([
        _resp("length", tool_calls=None, content="prose"),
        _resp("length", tool_calls=None, content="still prose"),
    ])
    m = _bare_model(client)
    m._query([{"role": "user", "content": "fix"}])   # arms
    m._query([{"role": "user", "content": "fix"}])   # recovery fails
    assert m._recovery_armed is False  # not re-armed; format-error policy takes over
