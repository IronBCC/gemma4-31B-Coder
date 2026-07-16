"""Direct-vLLM model class for mini-SWE-agent.

This mirrors the remote eval path the smoke used, but keeps the request clamps
local so the same logic can be versioned with the repo.

One-shot recovery mode (2026-07-15): raw base resolves 16/30 but produces
`RepeatedFormatError` on ~4/30 where a turn finishes on `length` with no tool
call (model consumes the 4k budget reasoning, never emits an action). Rather
than raise the global clamp (32k caused runaway thought loops) or retrain, we
patch the CLIENT: when a normal response is length+no-tool AND text-salvage
also fails, arm recovery for the NEXT harness retry of that step. The recovery
request forces a structured bash call (max_tokens 1024, tool_choice=bash,
thinking off, terse instruction). Success clears recovery; failure falls through
to the existing format-error policy. Normal behavior for the 26 passing cases is
untouched.
"""
from __future__ import annotations

import json
import time
from typing import Any

from openai import OpenAI
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function

from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall import BASH_TOOL

try:
    from model_clamps import (
        add_budget_pressure_messages,
        build_call_kwargs,
        compact_live_messages,
        extract_recoverable_command_from_arguments,
        extract_recoverable_command_from_text,
        forced_command_for_history,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback for package-style imports
    from phaseH_eval.model_clamps import (
        add_budget_pressure_messages,
        build_call_kwargs,
        compact_live_messages,
        extract_recoverable_command_from_arguments,
        extract_recoverable_command_from_text,
        forced_command_for_history,
    )

RECOVERY_MAX_TOKENS = 1024
RECOVERY_INSTRUCTION = "Emit exactly one bash tool call now. No explanation."


def response_needs_recovery(response: Any) -> bool:
    """True when a turn finished on length with no usable tool call — the
    RepeatedFormatError signature. Called AFTER normalize/salvage, so a
    text-salvaged command already cleared the no-tool condition."""
    choice = response.choices[0]
    finished_on_length = getattr(choice, "finish_reason", None) == "length"
    has_tool_call = bool(getattr(choice.message, "tool_calls", None))
    return finished_on_length and not has_tool_call


def build_recovery_kwargs(base_kwargs: dict) -> tuple[dict, dict]:
    """Return (call_kwargs, create_kwargs) forcing a single structured bash call
    with thinking disabled and a tight token budget."""
    call_kwargs = dict(base_kwargs)
    call_kwargs.pop("extra_body", None)
    call_kwargs["max_tokens"] = RECOVERY_MAX_TOKENS
    call_kwargs["tool_choice"] = {"type": "function", "function": {"name": "bash"}}
    create_kwargs = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    return call_kwargs, create_kwargs


class VllmDirectModel(LitellmModel):
    abort_exceptions: list[type[Exception]] = [KeyboardInterrupt]

    def __init__(self, *, config_class: Any = LitellmModelConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)
        mk = dict(self.config.model_kwargs)
        base_url = mk.pop("api_base", None) or "http://localhost:8010/v1"
        api_key = mk.pop("api_key", None) or "dummy"
        for key in ("drop_params", "custom_llm_provider", "api_version", "model_list", "mock_response",
                    "litellm_model_registry"):
            mk.pop(key, None)
        self._call_kwargs = build_call_kwargs(mk)
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._recovery_armed = False

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        forced_command = forced_command_for_history(messages)
        if forced_command:
            return _synthetic_bash_message(forced_command)
        return super().query(messages, **kwargs)

    def _query(self, messages: list[dict[str, str]], **kwargs):
        served = self.config.model_name.split("/", 1)[-1]
        call_kwargs = self._call_kwargs | kwargs
        # NOTE: Gemma-4 thinking is handled by the server's chat template; not
        # forced here. See module docstring for the recovery rationale.
        recovering = self._recovery_armed
        if recovering:
            self._recovery_armed = False  # one-shot: consume the arm now
            call_kwargs, create_kwargs = build_recovery_kwargs(call_kwargs)
            live = compact_live_messages(add_budget_pressure_messages(messages))
            live = live + [{"role": "user", "content": RECOVERY_INSTRUCTION}]
        else:
            extra_body = call_kwargs.pop("extra_body", None)
            create_kwargs = {"extra_body": extra_body} if extra_body else {}
            live = compact_live_messages(add_budget_pressure_messages(messages))
        response = self._client.chat.completions.create(
            model=served,
            messages=live,
            tools=[BASH_TOOL],
            **create_kwargs,
            **call_kwargs,
        )
        normalize_bash_tool_calls(response)
        # Arm recovery for the NEXT retry only on a normal (non-recovery) request
        # that finished on length with no salvageable tool call. A recovery
        # request that still fails falls through to the format-error policy.
        if not recovering and response_needs_recovery(response):
            self._recovery_armed = True
        return response

    def _calculate_cost(self, response) -> dict[str, float]:
        return {"cost": 0.0}


def _synthetic_bash_message(command: str) -> dict:
    tool_call_id = f"guarded-tool-{int(time.time() * 1000)}"
    tool_call = {
        "id": tool_call_id,
        "function": {"arguments": json.dumps({"command": command}), "name": "bash"},
        "type": "function",
    }
    return {
        "content": None,
        "role": "assistant",
        "tool_calls": [tool_call],
        "extra": {
            "actions": [{"command": command, "tool_call_id": tool_call_id}],
            "response": {"guarded_forced_command": command},
            "cost": 0.0,
            "timestamp": time.time(),
        },
    }


def normalize_bash_tool_calls(response: Any) -> None:
    """Coerce recoverable malformed bash tool calls to mini-SWE's schema."""

    message = response.choices[0].message
    tool_calls = list(message.tool_calls or [])
    if not tool_calls:
        command = extract_recoverable_command_from_text(message.content or "")
        if command:
            message.tool_calls = [
                ChatCompletionMessageToolCall(
                    id=f"salvaged-tool-{int(time.time() * 1000)}",
                    function=Function(arguments=json.dumps({"command": command}), name="bash"),
                    type="function",
                )
            ]
            message.content = None
        return

    for tool_call in tool_calls:
        function = getattr(tool_call, "function", None)
        if function is None or getattr(function, "name", None) != "bash":
            continue
        command = extract_recoverable_command_from_arguments(getattr(function, "arguments", None))
        if command:
            function.arguments = json.dumps({"command": command})
