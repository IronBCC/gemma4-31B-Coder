"""Direct-vLLM model class for mini-SWE-agent.

This mirrors the remote eval path the smoke used, but keeps the request clamps
local so the same logic can be versioned with the repo.
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

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        forced_command = forced_command_for_history(messages)
        if forced_command:
            return _synthetic_bash_message(forced_command)
        return super().query(messages, **kwargs)

    def _query(self, messages: list[dict[str, str]], **kwargs):
        served = self.config.model_name.split("/", 1)[-1]
        call_kwargs = self._call_kwargs | kwargs
        # NOTE: Gemma-4 thinking (chat_template enable_thinking / <|channel>thought)
        # is NOT enabled here. Probe on port 8012 (2026-07-12) proved this v6 adapter
        # was not trained for the thinking channel: enable_thinking=true yields
        # degenerate "<|turn>model" loops and no tool call, while thinking-off yields
        # clean bash tool calls. Thinking support requires retraining with that
        # channel format, not a serving flag. Optional opt-in kept via extra_body.
        extra_body = call_kwargs.pop("extra_body", None)
        create_kwargs = {"extra_body": extra_body} if extra_body else {}
        response = self._client.chat.completions.create(
            model=served,
            messages=compact_live_messages(add_budget_pressure_messages(messages)),
            tools=[BASH_TOOL],
            **create_kwargs,
            **call_kwargs,
        )
        normalize_bash_tool_calls(response)
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
