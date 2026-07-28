#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

import logging
import os
import sys
import unittest
from typing import Any, Dict, List

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

llm_agent = {
    "agent_name": os.getenv("agent_name", "openai_completions_test"),
    "agent_description": os.getenv(
        "agent_description", "This is a test agent for OpenAICompletionsEventHandler."
    ),
    "instructions": load_prompt_from_md(
        os.path.join(os.path.dirname(__file__), os.getenv("prompt_file", "prompt.md"))
    ),
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
        "max_tokens": "1024",
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


class OpenAICompletionsEventHandlerTest(unittest.TestCase):
    def setUp(self):
        self.agent = llm_agent
        self.handler = OpenAICompletionsEventHandler(logger, self.agent, **setting)
        logger.info("Initiate OpenAICompletionsEventHandlerTest ...")

    def tearDown(self):
        logger.info("Destroy OpenAICompletionsEventHandlerTest ...")
        if hasattr(self.handler, "close"):
            self.handler.close()

    @unittest.skip("interactive chatbot")
    def test_run_chatbot_non_streaming(self):
        logger.info("Starting chatbot in non-streaming usage mode...")

        while True:
            user_input = input("User: ")
            if user_input.strip().lower() in ["exit", "quit"]:
                logger.info("User requested exit. Stopping the chatbot.")
                print("Chatbot: Goodbye!")
                break

            self.handler.short_term_memory.append(
                {
                    "message": {"role": "user", "content": user_input},
                    "created_at": pendulum.now("UTC"),
                }
            )
            input_messages = get_input_messages(
                self.handler.short_term_memory,
                self.agent["num_of_messages"],
            )

            run_id = self.handler.ask_model(input_messages)

            logger.info(f"Non-streaming ask_model run ID: {run_id}")
            logger.info("Final response from model: %s", self.handler.final_output)

            self.handler.short_term_memory.append(
                {
                    "message": {
                        "role": self.handler.final_output["role"],
                        "content": self.handler.final_output["content"],
                    },
                    "created_at": pendulum.now("UTC"),
                }
            )

            print(f"Chatbot: {self.handler.final_output['content']}")

    # @unittest.skip("interactive chatbot")
    def test_run_chatbot_streaming(self):
        logger.info("Starting chatbot in streaming usage mode...")

        import threading
        from queue import Queue

        user_input = "Hello!"  # Initial prompt to start the conversation
        while True:
            stream_queue = Queue()
            stream_event = threading.Event()

            self.handler.short_term_memory.append(
                {
                    "message": {"role": "user", "content": user_input},
                    "created_at": pendulum.now("UTC"),
                }
            )
            input_messages = get_input_messages(
                self.handler.short_term_memory,
                self.agent["num_of_messages"],
            )

            stream_thread = threading.Thread(
                target=self.handler.ask_model,
                args=(input_messages, stream_queue, stream_event),
                daemon=True,
            )
            stream_thread.start()

            # Wait until we get the run_id from the queue
            current_run = stream_queue.get()
            if current_run["name"] == "run_id":
                logger.info(f"Current Run ID: {current_run['value']}")

            # Wait until streaming is done
            stream_event.wait()
            logger.info("Streaming ask_model finished.")

            logger.info("Final response from model: %s", self.handler.final_output)

            final = self.handler.final_output or {}
            if final.get("role") and final.get("content") is not None:
                self.handler.short_term_memory.append(
                    {
                        "message": {
                            "role": final["role"],
                            "content": final["content"],
                        },
                        "created_at": pendulum.now("UTC"),
                    }
                )
                print(f"Chatbot: {final['content']}")
            else:
                # ask_model failed (see logs for BadRequestError etc.).
                # Skip the memory append and let the user recover.
                print("Chatbot: [no response - the model call failed; see logs above]")

            user_input = input("User: ")
            if user_input.strip().lower() in ["exit", "quit"]:
                logger.info("User requested exit. Stopping the chatbot.")
                print("Chatbot: Goodbye!")
                break

    @unittest.skip("interactive chatbot")
    def test_ask_model_no_stream(self):
        logger.info("Start one-shot non-streaming usage ...")

        content = "What is the weather in Tokyo today?"
        input_messages: List[Dict[str, Any]] = [{"role": "user", "content": content}]

        run_id = self.handler.ask_model(input_messages)
        logger.info(f"Non-streaming ask_model run ID: {run_id}")
        logger.info("Final response from model: %s", self.handler.final_output)

    @unittest.skip("interactive chatbot")
    def test_ask_model_stream(self):
        logger.info("Start one-shot streaming usage ...")

        import threading
        from queue import Queue

        content = "What is the weather in Tokyo today?"
        input_messages: List[Dict[str, Any]] = [{"role": "user", "content": content}]

        stream_queue = Queue()
        stream_event = threading.Event()

        stream_thread = threading.Thread(
            target=self.handler.ask_model,
            args=(input_messages, stream_queue, stream_event),
            daemon=True,
        )
        stream_thread.start()

        # Wait until we get the run_id from the queue
        current_run = stream_queue.get()
        if current_run["name"] == "run_id":
            logger.info(f"Current Run ID: {current_run['value']}")

        # Wait until streaming is done
        stream_event.wait()
        logger.info("Streaming ask_model finished.")
        logger.info("Final response from model: %s", self.handler.final_output)


if __name__ == "__main__":
    unittest.main()
