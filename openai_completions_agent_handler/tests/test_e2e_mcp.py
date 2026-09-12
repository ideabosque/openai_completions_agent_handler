#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
End-to-end test: LLM → openai_completions_agent_handler → silvaengine_gateway (MCP)
→ mcp_protocol_plugin → mcp_skill_provider → harness_engineering_engine.

Uses the same pattern as test_chatbot.py: the handler discovers MCP tools
internally from the mcp_servers config. The gateway must already be running
(start with: python -m silvaengine_gateway.tests.run_daemon).

The system prompt is loaded from a markdown file specified by the `e2e_prompt_file`
env var (falls back to `prompt_file`, then to ingredient-research-agent.md). The MCP tools available depend
on what the gateway exposes — typically search_skills, get_skill, run_command,
and poll_command from mcp_skill_provider, but the test is generic and works
with any MCP tool the gateway provides.

Usage:
    # pytest:
    pytest openai_completions_agent_handler/tests/test_e2e_mcp.py -v

    # Standalone (streaming — default):
    python -m openai_completions_agent_handler.tests.test_e2e_mcp

    # Non-streaming:
    python -m openai_completions_agent_handler.tests.test_e2e_mcp --no-stream

    # Custom prompt:
    python -m openai_completions_agent_handler.tests.test_e2e_mcp --prompt "Search for ingredient research skills."

Requires .env in this directory (see .env.example). Uses the same model,
base_url, mcp_base_url, endpoint_id, and part_id settings as test_chatbot.py.
"""
from __future__ import print_function

__author__ = "bibow"

import argparse
import logging
import os
import sys
import threading
import unittest
from queue import Queue
from typing import Any, Dict, List

# Fix Windows console encoding for Unicode output (emoji, em-dashes, etc.)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pendulum
from dotenv import load_dotenv

# Load .env from the same directory as this test script.
load_dotenv()

setting = {
    "enable_timeline_log": os.getenv("enable_timeline_log", True),
}


def load_prompt_from_md(md_file_path):
    try:
        with open(md_file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {md_file_path}")


# Insert sibling packages onto sys.path so imports resolve when running standalone.
sys.path.insert(0, f"{os.getenv('base_dir')}/silvaengine_utility")
sys.path.insert(1, f"{os.getenv('base_dir')}/ai_agent_handler")
sys.path.insert(2, f"{os.getenv('base_dir')}/openai_completions_agent_handler")
sys.path.insert(3, f"{os.getenv('base_dir')}/mcp_http_client")

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

from openai_completions_agent_handler import OpenAICompletionsEventHandler


def _env_bool(name):
    v = os.getenv(name)
    if v is None:
        return None
    return v.strip().lower() in {"1", "true", "yes", "on"}


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

mcp_headers = {
    "Part-Id": os.getenv("part_id"),
    "Content-Type": "application/json",
}

if os.getenv("mcp_api_key") and not os.getenv("mcp_bearer_token"):
    mcp_headers["X-Api-Key"] = os.getenv("mcp_api_key")

if os.getenv("mcp_bearer_token") and not os.getenv("mcp_api_key"):
    mcp_headers["Authorization"] = f"Bearer {os.getenv('mcp_bearer_token')}"

# ── Load the system prompt from a markdown file ────────────────────────
# Uses `e2e_prompt_file` from .env (falls back to `prompt_file`, then to
# ingredient-research-agent.md). The file is resolved relative to this test
# directory, or if not found, relative to
# base_dir/ingredient_optimization_agent/.claude/agents/.
_prompt_file = os.getenv(
    "e2e_prompt_file",
    os.getenv("prompt_file", "ingredient-research-agent.md"),
)
_prompt_path = os.path.join(os.path.dirname(__file__), _prompt_file)

if not os.path.exists(_prompt_path):
    # Try resolving relative to base_dir (for prompts outside the tests dir)
    _alt_path = os.path.join(
        os.getenv("base_dir", ""),
        "ingredient_optimization_agent",
        ".claude",
        "agents",
        _prompt_file,
    )
    if os.path.exists(_alt_path):
        _prompt_path = _alt_path
    else:
        logger.warning(
            f"Prompt file '{_prompt_file}' not found at {_prompt_path} or {_alt_path}."
        )

_instructions = load_prompt_from_md(_prompt_path)

# Append tool usage instructions directing the LLM to use available MCP tools
_instructions += (
    "\n\n## Tool Usage Instructions\n"
    "You have access to MCP tools provided by the gateway: search_skills, get_skill, "
    "run_command, and poll_command.\n"
    "When asked to work with a skill:\n"
    "1. Use get_skill to retrieve the full skill instructions, including allowed_commands.\n"
    "2. Read and understand the skill instructions thoroughly.\n"
    "3. Follow the skill's workflow — use run_command to execute the allowlisted commands\n"
    "   described in the skill. The argv must match one of the skill's allowed_commands entries.\n"
    "4. If a command runs in the background (returns a run_id), use poll_command to check\n"
    "   its status until completed.\n"
    "5. Process the command output and incorporate it into your response.\n"
    "Always read the skill first, then execute commands as the skill directs.\n"
)

llm_agent = {
    "agent_name": os.getenv("agent_name", "e2e_mcp_test"),
    "agent_description": os.getenv(
        "agent_description",
        "End-to-end test agent: LLM → MCP gateway → mcp_skill_provider → "
        "harness_engineering_engine.",
    ),
    "instructions": _instructions,
    "llm": {"llm_name": "gpt"},
    "mcp_servers": [
        {
            "name": "internal_mcp_server",
            "setting": {
                "base_url": f"{os.getenv('mcp_base_url')}/{os.getenv('endpoint_id')}/mcp",
                "headers": mcp_headers,
            },
        },
    ],
    "configuration": {
        "base_url": os.getenv("base_url"),
        "model": os.getenv("model"),
        "openai_api_key": os.getenv("openai_api_key"),
        "temperature": "0.7",
        "max_completion_tokens": os.getenv(
            "max_completion_tokens", os.getenv("max_tokens", "8192")
        ),
        "instructions_role": os.getenv("instructions_role", "system"),
        "max_tool_call_depth": 8,
        "request_timeout_seconds": 120.0,
        "connect_timeout_seconds": 10.0,
        "max_retries": 2,
        "tools": [],
        "extra_body": extra_body,
        "enable_thinking": _env_bool("enable_thinking"),
        "separate_reasoning": _env_bool("separate_reasoning"),
        "reasoning_effort": os.getenv("reasoning_effort"),
        "debug_log_request_messages": _env_bool("debug_log_request_messages"),
    },
    "num_of_messages": 30,
    "tool_call_role": os.getenv("tool_call_role", "tool"),
}


def get_input_messages(
    messages: List[Dict[str, Any]], num_of_messages: int
) -> List[Dict[str, Any]]:
    """
    Convert the chat short_term_memory into a list of messages in the format
    expected by the OpenAI Chat Completions API.
    """
    return [
        msg["message"]
        for msg in sorted(messages, key=lambda x: x["created_at"], reverse=True)
    ][:num_of_messages][::-1]


class E2EMCPTest(unittest.TestCase):
    """End-to-end test: LLM → MCP gateway → mcp_skill_provider → harness_engineering_engine.

    The handler discovers MCP tools (search_skills, get_skill, run_command,
    poll_command) from the gateway's /mcp endpoint. A user prompt triggers the
    LLM to call these tools, routing through:
        gateway → mcp_protocol_plugin → mcp_skill_provider → GraphQL
        → harness_engineering_engine
    """

    def setUp(self):
        self.agent = llm_agent
        self.handler = OpenAICompletionsEventHandler(logger, self.agent, **setting)
        logger.info("Initiate E2EMCPTest ...")

    def tearDown(self):
        logger.info("Destroy E2EMCPTest ...")
        if hasattr(self.handler, "close"):
            self.handler.close()

    # ── Helpers ───────────────────────────────────────────────

    def _run_streaming(self, prompt):
        """Run ask_model in streaming mode and return final output dict."""
        self.handler.short_term_memory.append(
            {
                "message": {"role": "user", "content": prompt},
                "created_at": pendulum.now("UTC"),
            }
        )
        input_messages = get_input_messages(
            self.handler.short_term_memory,
            self.agent["num_of_messages"],
        )
        stream_queue = Queue()
        stream_event = threading.Event()
        stream_thread = threading.Thread(
            target=self.handler.ask_model,
            args=(input_messages, stream_queue, stream_event),
            daemon=True,
        )
        stream_thread.start()
        current_run = stream_queue.get()
        if current_run["name"] == "run_id":
            logger.info(f"Current Run ID: {current_run['value']}")
        stream_event.wait()
        logger.info("Streaming ask_model finished.")
        return self.handler.final_output or {}

    def _run_non_streaming(self, prompt):
        """Run ask_model in non-streaming mode and return final output dict."""
        self.handler.short_term_memory.append(
            {
                "message": {"role": "user", "content": prompt},
                "created_at": pendulum.now("UTC"),
            }
        )
        input_messages = get_input_messages(
            self.handler.short_term_memory,
            self.agent["num_of_messages"],
        )
        run_id = self.handler.ask_model(input_messages)
        logger.info(f"Non-streaming ask_model run ID: {run_id}")
        return self.handler.final_output or {}

    @staticmethod
    def _print_response(label, content, limit=3000):
        """Print a truncated LLM response with a header."""
        print(f"\n=== LLM Response ({label}) ===")
        print(content[:limit])
        if len(content) > limit:
            print(f"... ({len(content)} chars total)")

    # ── Streaming Tests ───────────────────────────────────────

    def test_01_search_skills_streaming(self):
        """LLM calls search_skills MCP tool (streaming mode)."""
        prompt = (
            "Search for skills related to 'ingredient research'. "
            "Use the search_skills tool to find matching skills, then "
            "summarize what you found."
        )
        final = self._run_streaming(prompt)
        logger.info("Final response from model: %s", final)
        content = final.get("content", "")
        self.assertTrue(content, "Final response is empty")
        self._print_response("search_skills, streaming", content, 2000)

    def test_02_get_skill_streaming(self):
        """LLM calls get_skill MCP tool (streaming mode)."""
        prompt = (
            "Get the full instructions for the 'ingredient-research' skill "
            "using the get_skill tool. Summarize the key sections of the "
            "skill instructions."
        )
        final = self._run_streaming(prompt)
        logger.info("Final response from model: %s", final)
        content = final.get("content", "")
        self.assertTrue(content, "Final response is empty")
        self._print_response("get_skill, streaming", content, 2000)

    def test_03_multi_tool_chain_streaming(self):
        """LLM chains search_skills → get_skill in a single turn (streaming)."""
        prompt = (
            "I need ingredient research capabilities. First, search for skills "
            "matching 'ingredient research'. Then, get the full instructions for "
            "the most relevant skill you found. Finally, summarize what the skill "
            "does and what commands it supports."
        )
        final = self._run_streaming(prompt)
        logger.info("Final response from model: %s", final)
        content = final.get("content", "")
        self.assertTrue(content, "Final response is empty")
        self._print_response("multi-tool chain, streaming", content, 3000)

    def test_04_get_skill_then_run_command_streaming(self):
        """LLM calls get_skill → run_command (streaming mode)."""
        prompt = (
            "Retrieve the 'ingredient-research' skill using get_skill. "
            "Read the skill instructions carefully, then follow the skill's "
            "workflow by using run_command to execute one of the allowlisted "
            "commands. Process the command output and summarize what the "
            "skill does and what the command produced."
        )
        final = self._run_streaming(prompt)
        logger.info("Final response from model: %s", final)
        content = final.get("content", "")
        self.assertTrue(content, "Final response is empty")
        self._print_response("get_skill + run_command, streaming", content, 3000)

    def test_05_run_command_background_then_poll_streaming(self):
        """LLM calls get_skill → run_command(background=true) → poll_command (streaming).

        Full round-trip:
            1. LLM calls get_skill("ingredient-research")
               → returns SKILL.md body + allowed_commands
            2. LLM calls run_command with background=true for a long-running command
               → returns run_id immediately
            3. LLM calls poll_command(run_id) to check status
               → returns status, stdout, stderr, exit_code
            4. LLM processes the polled output and produces final response
        """
        prompt = (
            "Retrieve the 'ingredient-research' skill using get_skill. "
            "Read the skill instructions carefully, then follow the skill's "
            "workflow by using run_command with background=true to execute "
            "one of the allowlisted commands. Then use poll_command to check "
            "the status of the background command until it completes. "
            "Process the command output and summarize what the skill does "
            "and what the command produced."
        )
        final = self._run_streaming(prompt)
        logger.info("Final response from model: %s", final)
        content = final.get("content", "")
        self.assertTrue(content, "Final response is empty")
        self._print_response(
            "get_skill + run_command background + poll_command, streaming",
            content,
            3000,
        )

    # ── Non-Streaming Tests ───────────────────────────────────

    def test_06_search_skills_non_streaming(self):
        """LLM calls search_skills MCP tool (non-streaming mode)."""
        prompt = (
            "Search for skills related to 'ingredient research'. "
            "Use the search_skills tool to find matching skills, then "
            "summarize what you found."
        )
        final = self._run_non_streaming(prompt)
        logger.info("Final response from model: %s", final)
        content = final.get("content", "")
        self.assertTrue(content, "Final response is empty")
        self._print_response("search_skills, non-streaming", content, 2000)

    def test_07_get_skill_non_streaming(self):
        """LLM calls get_skill MCP tool (non-streaming mode)."""
        prompt = (
            "Get the full instructions for the 'ingredient-research' skill "
            "using the get_skill tool. Summarize the key sections."
        )
        final = self._run_non_streaming(prompt)
        logger.info("Final response from model: %s", final)
        content = final.get("content", "")
        self.assertTrue(content, "Final response is empty")
        self._print_response("get_skill, non-streaming", content, 2000)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end test: LLM → MCP → mcp_skill_provider → "
            "harness_engineering_engine"
        )
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Use non-streaming mode for the one-shot test",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt for one-shot mode",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.prompt:
        # One-shot mode: send a single prompt and print the result
        print(f"\n>>> Prompt: {args.prompt}\n")

        handler = OpenAICompletionsEventHandler(logger, llm_agent, **setting)
        handler.short_term_memory.append(
            {
                "message": {"role": "user", "content": args.prompt},
                "created_at": pendulum.now("UTC"),
            }
        )
        input_messages = get_input_messages(
            handler.short_term_memory,
            llm_agent["num_of_messages"],
        )

        if args.no_stream:
            run_id = handler.ask_model(input_messages)
            print(f"\nRun ID: {run_id}")
        else:
            stream_queue = Queue()
            stream_event = threading.Event()
            stream_thread = threading.Thread(
                target=handler.ask_model,
                args=(input_messages, stream_queue, stream_event),
                daemon=True,
            )
            stream_thread.start()
            current_run = stream_queue.get()
            if current_run["name"] == "run_id":
                print(f"\nRun ID: {current_run['value']}")
            stream_event.wait()
            stream_thread.join(timeout=5)

        final = handler.final_output or {}
        print(f"\n=== Final Response ===")
        print(final.get("content", "<no content>"))
        if hasattr(handler, "close"):
            handler.close()
    else:
        unittest.main()