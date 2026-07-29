#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Standalone script exercising one complete tool-call round-trip.

Default is streaming. Pass --no-stream to use the non-streaming path.

Flow:
    user prompt -> model -> tool_call -> local Python function -> tool result
                                                              -> model -> final text

Run:
    python -m openai_completions_agent_handler.tests.test_tool_call
    python -m openai_completions_agent_handler.tests.test_tool_call --no-stream
    python -m openai_completions_agent_handler.tests.test_tool_call "Weather in Singapore?"
    python -m openai_completions_agent_handler.tests.test_tool_call --no-stream "Weather in Singapore?"

Requires .env in this directory (see .env.example). Uses the same model and
base_url settings as test_chatbot.py. Override `instructions_role=user` in .env
for Gemma family models.
"""

from __future__ import print_function

__author__ = "bibow"

import logging
import os
import sys
import threading
from queue import Queue
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()

setting = {
    "enable_timeline_log": os.getenv("enable_timeline_log", False),
}


def get_weather(city: str) -> Dict[str, Any]:
    """Local stand-in for a real weather API."""
    canned = {
        "Tokyo": {"temperature_c": 18, "conditions": "Cloudy", "humidity_pct": 72},
        "San Francisco": {
            "temperature_c": 14,
            "conditions": "Foggy",
            "humidity_pct": 88,
        },
        "Singapore": {
            "temperature_c": 30,
            "conditions": "Thunderstorms",
            "humidity_pct": 85,
        },
    }
    data = canned.get(
        city, {"temperature_c": 20, "conditions": "Clear", "humidity_pct": 50}
    )
    return {"city": city, **data}


LOCAL_TOOLS = {
    "get_weather": get_weather,
}


sys.path.insert(0, f"{os.getenv('base_dir')}/silvaengine_utility")
sys.path.insert(1, f"{os.getenv('base_dir')}/ai_agent_handler")
sys.path.insert(2, f"{os.getenv('base_dir')}/openai_completions_agent_handler")

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

from ai_agent_handler import AIAgentEventHandler

# This smoke test executes tools locally and does not need the deployment
# message-stream invoker, which imports the full core-engine dependency tree.
AIAgentEventHandler._initialize_message_invoker = lambda self, logger, setting: None

from openai_completions_agent_handler import OpenAICompletionsEventHandler


def _env_bool(name):
    v = os.getenv(name)
    if v is None:
        return None
    return v.strip().lower() in {"1", "true", "yes", "on"}


MODEL = os.getenv("model") or ""
extra_body = None
if os.getenv("thinking_type"):
    extra_body = {
        "thinking": {
            "type": os.getenv("thinking_type"),
            "clear_thinking": (
                _env_bool("clear_thinking")
                if _env_bool("clear_thinking") is not None
                else False
            ),
        }
    }


llm_agent = {
    "agent_name": "openai_completions_tool_call_demo",
    "agent_description": "Demo agent that exercises one tool call end-to-end.",
    "instructions": (
        "You are a concise weather assistant. When a user asks about weather, "
        "call the `get_weather` tool with the city name. After receiving the "
        "tool result, reply in one short sentence."
    ),
    "llm": {"llm_name": "gpt"},
    "configuration": {
        "base_url": os.getenv("base_url"),
        "model": MODEL,
        "openai_api_key": os.getenv("openai_api_key"),
        "temperature": "0.2",
        "max_completion_tokens": os.getenv("max_completion_tokens", os.getenv("max_tokens", "8192")),
        "instructions_role": os.getenv("instructions_role", "system"),
        "max_tool_call_depth": 4,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name, e.g. Tokyo",
                            }
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "extra_body": extra_body,
        "enable_thinking": _env_bool("enable_thinking"),
        "separate_reasoning": _env_bool("separate_reasoning"),
        "reasoning_effort": os.getenv("reasoning_effort"),
    },
    "num_of_messages": 30,
    "tool_call_role": "developer",
}


def make_logging_dispatcher(tools, counter):
    """Wrap each tool so every invocation prints a clear [TOOL CALL]/[RESULT] line."""

    def dispatcher(name):
        fn = tools.get(name)
        if fn is None:
            print(f"[TOOL MISS] model requested unknown tool: {name}")
            return None

        def wrapped(**kwargs):
            counter["calls"] += 1
            print(f"\n[TOOL CALL #{counter['calls']}] {name}({kwargs})")
            result = fn(**kwargs)
            print(f"[TOOL RESULT  #{counter['calls']}] {result}\n")
            return result

        return wrapped

    return dispatcher


def _run_streaming(handler, input_messages):
    stream_queue = Queue()
    stream_event = threading.Event()
    thread = threading.Thread(
        target=handler.ask_model,
        args=(input_messages, stream_queue, stream_event),
        daemon=True,
    )
    thread.start()

    # First message on the queue is always the run_id from the first chunk.
    first = stream_queue.get()
    run_id = first.get("value") if first.get("name") == "run_id" else None
    print(f">>> Streaming (Run ID: {run_id})")
    print()  # blank line before the live chunk stream
    # handle_stream's print(delta.content, end='', flush=True) renders chunks
    # token-by-token to stdout; we only wait for the completion signal here.

    stream_event.wait()
    print()  # newline after the streamed chunks
    return run_id


def _run_non_streaming(handler, input_messages):
    return handler.ask_model(input_messages)


def main(
    prompt: str = "What is the weather in Tokyo right now?", stream: bool = True
) -> None:
    with OpenAICompletionsEventHandler(logger, llm_agent, **setting) as handler:
        handler.short_term_memory = []
        # Resolve tool calls against the local function table instead of MCP.
        counter = {"calls": 0}
        handler.get_function = make_logging_dispatcher(LOCAL_TOOLS, counter)
        handler.invoke_async_funct = lambda **kwargs: None

        print(f"\n>>> User: {prompt}\n")
        input_messages = [{"role": "user", "content": prompt}]

        if stream:
            run_id = _run_streaming(handler, input_messages)
        else:
            run_id = _run_non_streaming(handler, input_messages)

        print("\n=== Diagnostics ===")
        print(f"mode              : {'streaming' if stream else 'non-streaming'}")
        requests = []
        if os.getenv("reasoning_effort"):
            requests.append(f"reasoning_effort={os.getenv('reasoning_effort')}")
        if os.getenv("thinking_type"):
            requests.append(f"thinking.type={os.getenv('thinking_type')}")
        if _env_bool("enable_thinking") is not None:
            requests.append(f"enable_thinking={_env_bool('enable_thinking')}")
        print(f"reasoning request : {', '.join(requests) or '<none>'}")
        print(f"run_id            : {run_id}")
        print(f"tool calls fired  : {counter['calls']}")
        print(f"final finish_reason: {handler._last_finish_reason}")
        if handler._last_usage:
            print(f"tokens            : {handler._last_usage}")

        final = handler.final_output or {}
        print(f"reasoning captured: {bool(final.get('reasoning_summary'))}")
        if final.get("reasoning_summary"):
            print("\n=== Reasoning Summary ===")
            print(final["reasoning_summary"])
        print("\n=== Final Response ===")
        print(final.get("content") or "<no content>")
        if final.get("truncated"):
            print("[!] Output was truncated (finish_reason=length)")
        if final.get("filtered"):
            print("[!] Output was filtered (finish_reason=content_filter)")

        if counter["calls"] == 0:
            print(
                "\n[diagnosis] No tool was invoked. The model returned text without "
                "emitting `tool_calls`. Likely causes:\n"
                "  - model doesn't support OpenAI-style function calling in this serving stack\n"
                "  - tool_choice was overridden to 'none' somewhere\n"
                "  - prompt didn't make a tool call obviously necessary (try a sharper ask)\n"
                "  - small instruction-tuned models often skip tool calls; try a larger one"
            )


if __name__ == "__main__":
    args = sys.argv[1:]
    stream_mode = "--no-stream" not in args
    prompt_args = [a for a in args if a != "--no-stream"]
    prompt = " ".join(prompt_args) or "What is the weather in Tokyo right now?"
    main(prompt, stream=stream_mode)
