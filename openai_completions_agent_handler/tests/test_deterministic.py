#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

__author__ = "bibow"

import json
import logging
import threading
import unittest
from queue import Queue
from unittest.mock import MagicMock, PropertyMock, patch

from openai_completions_agent_handler.openai_completions_agent_handler import (
    OpenAICompletionsEventHandler,
    ToolCallDepthExceeded,
    _omit_none,
    _ThinkTagSplitter,
)

# Prevent the base class from attempting to resolve ai_agent_core_engine during tests.
from ai_agent_handler import AIAgentEventHandler

patch.object(
    AIAgentEventHandler, "_initialize_message_invoker", lambda self, logger, setting: None
).start()


def _make_handler(agent_overrides=None, config_overrides=None):
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    agent = {
        "instructions": "You are a helpful assistant.",
        "llm": {"llm_name": "gpt"},
        "configuration": {
            "model": "gpt-4o",
            "openai_api_key": "sk-test-key-value",
            "temperature": "0.8",
            "max_tokens": "1000",
        },
        "num_of_messages": 30,
        "tool_call_role": "developer",
    }
    if config_overrides:
        agent["configuration"].update(config_overrides)
    if agent_overrides:
        agent.update(agent_overrides)
    setting = {
        "region_name": "us-east-1",
        "aws_access_key_id": "test",
        "aws_secret_access_key": "test",
        "funct_bucket_name": "test-bucket",
        "funct_zip_path": "/tmp",
        "funct_extract_path": "/tmp/extract",
    }
    return logger, agent, setting


class TestOmitNone(unittest.TestCase):
    def test_removes_none_values(self):
        self.assertEqual(_omit_none({"a": 1, "b": None, "c": 0}), {"a": 1, "c": 0})

    def test_keeps_falsy_non_none(self):
        self.assertEqual(_omit_none({"a": 0, "b": "", "c": False, "d": []}), {"a": 0, "b": "", "c": False, "d": []})


class TestInit(unittest.TestCase):
    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_1_creates_client_with_config(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        self.assertIsNotNone(handler.client)
        self.assertEqual(handler.model_setting["model"], "gpt-4o")
        self.assertEqual(handler.model_setting["temperature"], 0.8)
        self.assertEqual(handler.model_setting["max_tokens"], 1000)

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_2_instructions_role_default_system(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        self.assertEqual(handler.instructions_role, "system")

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_3_instructions_role_developer(self, mock_openai):
        logger, agent, setting = _make_handler(config_overrides={"instructions_role": "developer"})
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        self.assertEqual(handler.instructions_role, "developer")

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_4_max_completion_tokens_preferred(self, mock_openai):
        logger, agent, setting = _make_handler(
            config_overrides={"max_tokens": "500", "max_completion_tokens": "1000"}
        )
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        self.assertNotIn("max_tokens", handler.model_setting)
        self.assertEqual(handler.model_setting["max_completion_tokens"], 1000)

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_5_enabled_tools_filtered(self, mock_openai):
        config = {
            "tools": [
                {"type": "function", "function": {"name": "a", "parameters": {}}},
                {"type": "function", "function": {"name": "b", "parameters": {}}},
            ],
            "enabled_tools": ["a"],
        }
        logger, agent, setting = _make_handler(config_overrides=config)
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        tool_names = [t["function"]["name"] for t in handler.model_setting.get("tools", [])]
        self.assertEqual(tool_names, ["a"])

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_6_close_releases_http_client(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        self.assertIsNotNone(handler._http_client)
        handler.close()
        self.assertIsNone(handler._http_client)

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_7_context_manager(self, mock_openai):
        logger, agent, setting = _make_handler()
        with OpenAICompletionsEventHandler(logger, agent, **setting) as handler:
            self.assertIsNotNone(handler._http_client)
        self.assertIsNone(handler._http_client)


class TestDepthLifecycle(unittest.TestCase):
    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_8_depth_returns_to_zero_on_success(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.role = "assistant"
        mock_response.id = "resp_1"
        handler.client.chat.completions.create.return_value = mock_response
        handler.ask_model([{"role": "user", "content": "Hi"}])
        self.assertEqual(handler._ask_model_depth, 0)

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_9_depth_returns_to_zero_on_exception(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.client = MagicMock()
        handler.client.chat.completions.create.side_effect = Exception("boom")
        with self.assertRaises(Exception):
            handler.ask_model([{"role": "user", "content": "Hi"}])
        self.assertEqual(handler._ask_model_depth, 0)

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_10_tool_call_depth_exceeded(self, mock_openai):
        logger, agent, setting = _make_handler(config_overrides={"max_tool_call_depth": "1"})
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.client = MagicMock()
        handler._ask_model_depth = 2
        result = handler.ask_model([{"role": "user", "content": "Hi"}])
        self.assertIn("error", handler.final_output.get("content", ""))
        self.assertEqual(handler._ask_model_depth, 2)


class TestNonStreamingResponse(unittest.TestCase):
    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_11_text_response_populates_final_output(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.client = MagicMock()
        msg = MagicMock()
        msg.content = "Hello world"
        msg.tool_calls = None
        msg.role = "assistant"
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = msg
        mock_response.choices[0].finish_reason = "stop"
        mock_response.id = "resp_1"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        run_id = handler.handle_response(mock_response, [{"role": "user", "content": "Hi"}])
        self.assertEqual(handler.final_output["content"], "Hello world")
        self.assertEqual(handler.final_output["role"], "assistant")
        self.assertIsNotNone(handler._last_usage)

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_12_tool_response_triggers_execution(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.client = MagicMock()
        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "get_weather"
        tool_call.function.arguments = '{"city": "NYC"}'
        msg = MagicMock()
        msg.content = None
        msg.tool_calls = [tool_call]
        msg.role = "assistant"
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = msg
        mock_response.choices[0].finish_reason = "tool_calls"
        mock_response.id = "resp_2"
        handler.get_function = MagicMock(return_value=lambda city: "72F")
        handler.invoke_async_funct = MagicMock()
        handler.handle_response(mock_response, [{"role": "user", "content": "Weather?"}])
        self.assertTrue(handler._tool_loop_continue)

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_13_empty_response_retry_bounded(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.client = MagicMock()
        msg = MagicMock()
        msg.content = ""
        msg.tool_calls = None
        msg.role = "assistant"
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = msg
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = None
        mock_response.id = "resp_3"
        with self.assertRaises(Exception):
            handler.handle_response(mock_response, [{"role": "user", "content": "Hi"}], retry_count=3)

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_14_finish_reason_length_marks_truncated(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.client = MagicMock()
        msg = MagicMock()
        msg.content = "partial"
        msg.tool_calls = None
        msg.role = "assistant"
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = msg
        mock_response.choices[0].finish_reason = "length"
        mock_response.id = "resp_4"
        mock_response.usage = None
        handler.handle_response(mock_response, [{"role": "user", "content": "Hi"}])
        self.assertTrue(handler.final_output.get("truncated"))


    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_14b_finish_reason_length_with_reasoning_only_uses_visible_fallback(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.client = MagicMock()
        msg = MagicMock()
        msg.content = ""
        msg.tool_calls = None
        msg.role = "assistant"
        msg.reasoning_content = "thinking without final text"
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = msg
        mock_response.choices[0].finish_reason = "length"
        mock_response.id = "resp_4b"
        mock_response.usage = None
        handler.handle_response(mock_response, [{"role": "user", "content": "Hi"}])
        self.assertTrue(handler.final_output.get("truncated"))
        self.assertEqual(
            handler.final_output["reasoning_summary"], "thinking without final text"
        )
        self.assertIn("truncated", handler.final_output["content"])

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_15_finish_reason_content_filter_marks_filtered(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.client = MagicMock()
        msg = MagicMock()
        msg.content = "filtered content"
        msg.tool_calls = None
        msg.role = "assistant"
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = msg
        mock_response.choices[0].finish_reason = "content_filter"
        mock_response.id = "resp_5"
        mock_response.usage = None
        handler.handle_response(mock_response, [{"role": "user", "content": "Hi"}])
        self.assertTrue(handler.final_output.get("filtered"))


class TestStreaming(unittest.TestCase):
    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_16_stream_text_accumulates(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)

        chunk1 = MagicMock()
        chunk1.id = "s_1"
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"
        chunk1.choices[0].delta.tool_calls = None
        chunk1.choices[0].finish_reason = None

        chunk2 = MagicMock()
        chunk2.id = "s_1"
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " world"
        chunk2.choices[0].delta.tool_calls = None
        chunk2.choices[0].finish_reason = "stop"

        stream = [chunk1, chunk2]
        messages = [{"role": "user", "content": "Hi"}]
        queue = Queue()
        event = threading.Event()

        run_id = handler.handle_stream(stream, messages, queue=queue, stream_event=event)
        self.assertEqual(run_id, "s_1")
        self.assertEqual(handler.final_output["content"], "Hello world")
        self.assertTrue(event.is_set())

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_17_stream_usage_only_chunk(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)

        text_chunk = MagicMock()
        text_chunk.id = "s_2"
        text_chunk.choices = [MagicMock()]
        text_chunk.choices[0].delta.content = "Hi"
        text_chunk.choices[0].delta.tool_calls = None
        text_chunk.choices[0].finish_reason = "stop"

        usage_chunk = MagicMock()
        usage_chunk.id = "s_2"
        usage_chunk.choices = []
        usage_chunk.usage = MagicMock(prompt_tokens=5, completion_tokens=1, total_tokens=6)

        stream = [text_chunk, usage_chunk]
        messages = [{"role": "user", "content": "Hi"}]
        event = threading.Event()
        handler.handle_stream(stream, messages, stream_event=event)
        self.assertIsNotNone(handler._last_usage)

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_18_stream_tool_calls_stitch_by_index(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)

        tc_chunk1 = MagicMock()
        tc_chunk1.id = "s_3"
        tc_chunk1.choices = [MagicMock()]
        tc_chunk1.choices[0].delta.content = None
        delta_tc = MagicMock()
        delta_tc.index = 0
        delta_tc.id = "call_a"
        delta_tc.function = MagicMock()
        delta_tc.function.name = "get_weather"
        delta_tc.function.arguments = None
        tc_chunk1.choices[0].delta.tool_calls = [delta_tc]
        tc_chunk1.choices[0].delta.content = None
        tc_chunk1.choices[0].finish_reason = None

        tc_chunk2 = MagicMock()
        tc_chunk2.id = "s_3"
        tc_chunk2.choices = [MagicMock()]
        tc_chunk2.choices[0].delta.content = None
        delta_tc2 = MagicMock()
        delta_tc2.index = 0
        delta_tc2.id = None
        delta_tc2.function = MagicMock()
        delta_tc2.function.name = None
        delta_tc2.function.arguments = '{"city": "NYC"}'
        tc_chunk2.choices[0].delta.tool_calls = [delta_tc2]
        tc_chunk2.choices[0].delta.content = None
        tc_chunk2.choices[0].finish_reason = "tool_calls"

        stream = [tc_chunk1, tc_chunk2]
        messages = [{"role": "user", "content": "Weather?"}]
        handler.get_function = MagicMock(return_value=lambda city: "72F")
        handler.invoke_async_funct = MagicMock()
        handler.handle_stream(stream, messages)
        self.assertTrue(handler._tool_loop_continue)

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_19_stream_finish_length_marks_truncated(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)

        chunk = MagicMock()
        chunk.id = "s_4"
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "partial"
        chunk.choices[0].delta.tool_calls = None
        chunk.choices[0].finish_reason = "length"

        handler.handle_stream([chunk], [{"role": "user", "content": "Hi"}])
        self.assertTrue(handler.final_output.get("truncated"))



    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_19b_stream_finish_length_with_reasoning_only_uses_visible_fallback(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)

        chunk = MagicMock()
        chunk.id = "s_4b"
        chunk.choices = [MagicMock()]
        delta = MagicMock()
        delta.content = None
        delta.tool_calls = None
        delta.reasoning_content = "thinking without final text"
        chunk.choices[0].delta = delta
        chunk.choices[0].finish_reason = "length"

        handler.handle_stream([chunk], [{"role": "user", "content": "Hi"}])
        self.assertTrue(handler.final_output.get("truncated"))
        self.assertEqual(
            handler.final_output["reasoning_summary"], "thinking without final text"
        )
        self.assertIn("truncated", handler.final_output["content"])


class TestConversationHistory(unittest.TestCase):
    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_20_assistant_tool_calls_content_none(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        messages = [{"role": "user", "content": "Hi"}]
        handler._append_assistant_with_tool_calls(
            [{"id": "call_1", "name": "test", "arguments": "{}", "type": "function"}],
            messages,
        )
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertIsNone(messages[1]["content"])
        self.assertEqual(messages[1]["tool_calls"][0]["id"], "call_1")

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_21_tool_result_message_format(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        messages = [{"role": "user", "content": "Hi"}]
        handler._append_assistant_with_tool_calls(
            [{"id": "call_1", "name": "test", "arguments": "{}", "type": "function"}],
            messages,
        )
        handler._append_tool_result("call_1", '"ok"', messages)
        self.assertEqual(messages[2]["role"], "tool")
        self.assertEqual(messages[2]["tool_call_id"], "call_1")
        self.assertIsInstance(messages[2]["content"], str)


class TestMessageTrimming(unittest.TestCase):
    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_22_drops_orphan_tool_message(self, mock_openai):
        logger, agent, setting = _make_handler(agent_overrides={"num_of_messages": 2})
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        messages = [
            {"role": "system", "content": "Helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "{}"},
        ]
        trimmed = handler._trim_messages_for_recursion(messages)
        self.assertNotEqual(trimmed[0].get("role"), "tool")

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_23_preserves_instruction_head(self, mock_openai):
        logger, agent, setting = _make_handler(agent_overrides={"num_of_messages": 2})
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        messages = [
            {"role": "system", "content": "Helpful."},
            {"role": "user", "content": "A"},
            {"role": "user", "content": "B"},
            {"role": "user", "content": "C"},
        ]
        trimmed = handler._trim_messages_for_recursion(messages)
        self.assertEqual(trimmed[0]["role"], "system")
        self.assertEqual(len(trimmed), 3)


class TestProviderExtensions(unittest.TestCase):
    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_26_response_format_passed_through(self, mock_openai):
        logger, agent, setting = _make_handler(
            config_overrides={"response_format": {"type": "json_object"}}
        )
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        self.assertEqual(handler.model_setting["response_format"]["type"], "json_object")

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_27_reasoning_content_captured_when_present(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.client = MagicMock()
        msg = MagicMock()
        msg.content = "Answer"
        msg.tool_calls = None
        msg.role = "assistant"
        msg.reasoning_content = "I thought about it"
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = msg
        mock_response.choices[0].finish_reason = "stop"
        mock_response.id = "resp_rc"
        mock_response.usage = None
        handler.handle_response(mock_response, [{"role": "user", "content": "Hi"}])
        self.assertEqual(handler.final_output["reasoning_summary"], "I thought about it")

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_28_reasoning_content_absent_no_error(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.client = MagicMock()
        msg = MagicMock()
        msg.content = "Answer"
        msg.tool_calls = None
        msg.role = "assistant"
        del msg.reasoning_content
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = msg
        mock_response.choices[0].finish_reason = "stop"
        mock_response.id = "resp_nrc"
        mock_response.usage = None
        handler.handle_response(mock_response, [{"role": "user", "content": "Hi"}])
        self.assertNotIn("reasoning_summary", handler.final_output)


class TestPayloadHygiene(unittest.TestCase):
    def test_omit_none_removes_none(self):
        payload = {"model": "gpt-4o", "temperature": None, "max_tokens": 100, "stop": None}
        clean = _omit_none(payload)
        self.assertNotIn("temperature", clean)
        self.assertNotIn("stop", clean)
        self.assertIn("model", clean)
        self.assertIn("max_tokens", clean)


class TestApiKeyRedaction(unittest.TestCase):
    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_api_key_redacted_from_error(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        redacted = handler._redact_api_key("error with sk-test-key-value in message")
        self.assertNotIn("sk-test-key-value", redacted)
        self.assertIn("****", redacted)


class TestExceptionPropagation(unittest.TestCase):
    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_api_error_propagates_unchanged(self, mock_openai):
        from openai import APIError
        import httpx

        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.client = MagicMock()
        err = APIError(
            message="rate limit",
            request=httpx.Request("POST", "http://localhost"),
            body={"error": {"message": "rate limit"}},
        )
        handler.client.chat.completions.create.side_effect = err
        with self.assertRaises(APIError):
            handler.invoke_model(messages=[{"role": "user", "content": "Hi"}])

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_bad_request_error_propagates_unchanged(self, mock_openai):
        from openai import BadRequestError

        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.client = MagicMock()
        err = BadRequestError(
            message="invalid parameter",
            response=MagicMock(),
            body={"error": {"message": "invalid parameter"}},
        )
        handler.client.chat.completions.create.side_effect = err
        with self.assertRaises(BadRequestError):
            handler.invoke_model(messages=[{"role": "user", "content": "Hi"}])

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_generic_exception_wrapped(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.client = MagicMock()
        handler.client.chat.completions.create.side_effect = ValueError("boom")
        with self.assertRaises(Exception) as cm:
            handler.invoke_model(messages=[{"role": "user", "content": "Hi"}])
        self.assertIn("Failed to invoke model", str(cm.exception))


class TestAskModelExceptionPropagation(unittest.TestCase):
    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_api_error_propagates_unchanged(self, mock_openai):
        from openai import APIError
        import httpx

        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.client = MagicMock()
        err = APIError(
            message="rate limit",
            request=httpx.Request("POST", "http://localhost"),
            body={"error": {"message": "rate limit"}},
        )
        handler.client.chat.completions.create.side_effect = err
        with self.assertRaises(APIError):
            handler.ask_model([{"role": "user", "content": "Hi"}])

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_bad_request_error_propagates_unchanged(self, mock_openai):
        from openai import BadRequestError

        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.client = MagicMock()
        err = BadRequestError(
            message="invalid parameter",
            response=MagicMock(),
            body={"error": {"message": "invalid parameter"}},
        )
        handler.client.chat.completions.create.side_effect = err
        with self.assertRaises(BadRequestError):
            handler.ask_model([{"role": "user", "content": "Hi"}])

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_tool_call_depth_exceeded_returns_error_output(self, mock_openai):
        logger, agent, setting = _make_handler(config_overrides={"max_tool_call_depth": "0"})
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.client = MagicMock()
        result = handler.ask_model([{"role": "user", "content": "Hi"}])
        self.assertIn("error", handler.final_output.get("content", ""))
        self.assertIn("exceeds maximum", handler.final_output["content"])

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_generic_exception_wrapped(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.client = MagicMock()
        handler.client.chat.completions.create.side_effect = ValueError("boom")
        with self.assertRaises(Exception) as cm:
            handler.ask_model([{"role": "user", "content": "Hi"}])
        self.assertIn("Failed to process model request", str(cm.exception))


class TestImageContentArray(unittest.TestCase):
    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_build_image_message_with_text(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        msg = handler._build_image_message("https://example.com/img.png", "Describe this image.")
        self.assertEqual(msg["role"], "user")
        self.assertEqual(msg["content"][0]["type"], "text")
        self.assertEqual(msg["content"][0]["text"], "Describe this image.")
        self.assertEqual(msg["content"][1]["type"], "image_url")
        self.assertEqual(msg["content"][1]["image_url"]["url"], "https://example.com/img.png")

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_build_image_message_without_text(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        msg = handler._build_image_message("data:image/png;base64,iVBORw0KGgo=")
        self.assertEqual(msg["role"], "user")
        self.assertEqual(len(msg["content"]), 1)
        self.assertEqual(msg["content"][0]["type"], "image_url")

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_reject_unsupported_image_scheme(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        with self.assertRaises(ValueError) as cm:
            handler._build_image_message("ftp://example.com/img.png")
        self.assertIn("Unsupported image URL scheme", str(cm.exception))


class TestAssembleExtraBody(unittest.TestCase):
    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_shorthand_only_builds_extra_body(self, mock_openai):
        logger, agent, setting = _make_handler(
            config_overrides={"enable_thinking": True, "separate_reasoning": True}
        )
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        eb = handler.model_setting.get("extra_body")
        self.assertIsNotNone(eb)
        # SGLang/Qwen3 nested form
        self.assertEqual(eb["chat_template_kwargs"], {"enable_thinking": True})
        self.assertNotIn("thinking", eb)
        self.assertTrue(eb["separate_reasoning"])

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_glm_thinking_is_explicit_extra_body(self, mock_openai):
        logger, agent, setting = _make_handler(
            config_overrides={"extra_body": {"thinking": {"type": "enabled"}}}
        )
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        self.assertEqual(
            handler.model_setting["extra_body"]["thinking"], {"type": "enabled"}
        )

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_shorthand_excluded_from_model_setting(self, mock_openai):
        logger, agent, setting = _make_handler(
            config_overrides={"enable_thinking": True, "separate_reasoning": False}
        )
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        self.assertNotIn("enable_thinking", handler.model_setting)
        self.assertNotIn("separate_reasoning", handler.model_setting)

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_explicit_extra_body_wins_for_non_chat_template_keys(self, mock_openai):
        logger, agent, setting = _make_handler(
            config_overrides={
                "separate_reasoning": True,
                "extra_body": {"separate_reasoning": False, "custom_key": "x"},
            }
        )
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        eb = handler.model_setting["extra_body"]
        self.assertFalse(eb["separate_reasoning"])
        self.assertEqual(eb["custom_key"], "x")

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_chat_template_kwargs_deep_merge(self, mock_openai):
        logger, agent, setting = _make_handler(
            config_overrides={
                "enable_thinking": True,
                "extra_body": {"chat_template_kwargs": {"other_flag": True}},
            }
        )
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        ctk = handler.model_setting["extra_body"]["chat_template_kwargs"]
        self.assertTrue(ctk["enable_thinking"])
        self.assertTrue(ctk["other_flag"])

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_no_extra_body_when_neither_shorthand_set(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        self.assertNotIn("extra_body", handler.model_setting)


class TestMergeInstructionsIntoFirstUser(unittest.TestCase):
    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_merges_into_string_content(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        merged = handler._merge_instructions_into_first_user(
            [{"role": "user", "content": "what's the weather?"}], "Be concise."
        )
        self.assertEqual(merged[0]["content"], "Be concise.\n\nwhat's the weather?")

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_merges_into_list_content_at_front(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        original = [
            {"role": "user", "content": [{"type": "text", "text": "Describe."}]}
        ]
        merged = handler._merge_instructions_into_first_user(original, "Be concise.")
        self.assertEqual(merged[0]["content"][0]["type"], "text")
        self.assertEqual(merged[0]["content"][0]["text"], "Be concise.")
        self.assertEqual(merged[0]["content"][1]["text"], "Describe.")

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_inserts_user_message_when_none_present(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        merged = handler._merge_instructions_into_first_user(
            [{"role": "assistant", "content": "earlier"}], "Be concise."
        )
        self.assertEqual(merged[0]["role"], "user")
        self.assertEqual(merged[0]["content"], "Be concise.")
        self.assertEqual(merged[1]["role"], "assistant")

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_does_not_mutate_input(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        original = [{"role": "user", "content": "hi"}]
        handler._merge_instructions_into_first_user(original, "system prompt")
        self.assertEqual(original, [{"role": "user", "content": "hi"}])


class TestHandleFunctionCallsBatching(unittest.TestCase):
    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_parallel_calls_produce_single_assistant_message(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.get_function = MagicMock(return_value=lambda **kw: {"ok": True})
        handler.invoke_async_funct = MagicMock()

        messages = []
        handler.handle_function_calls(
            [
                {"id": "call_a", "function": {"name": "fa", "arguments": "{}"}},
                {"id": "call_b", "function": {"name": "fb", "arguments": "{}"}},
            ],
            messages,
        )

        # One assistant message with two tool_calls, then two tool results.
        self.assertEqual(messages[0]["role"], "assistant")
        self.assertIsNone(messages[0]["content"])
        self.assertEqual(len(messages[0]["tool_calls"]), 2)
        self.assertEqual(
            [tc["id"] for tc in messages[0]["tool_calls"]], ["call_a", "call_b"]
        )
        self.assertEqual(messages[1]["role"], "tool")
        self.assertEqual(messages[1]["tool_call_id"], "call_a")
        self.assertEqual(messages[2]["role"], "tool")
        self.assertEqual(messages[2]["tool_call_id"], "call_b")
        self.assertEqual(len(messages), 3)

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_reasoning_preserved_in_final_output_not_messages(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.get_function = MagicMock(return_value=lambda **kw: {"ok": True})
        handler.invoke_async_funct = MagicMock()
        handler.final_output["reasoning_summary"] = "planned tool use"
        messages = []

        handler.handle_function_calls(
            [{"id": "call_a", "function": {"name": "fa", "arguments": "{}"}}],
            messages,
        )

        self.assertNotIn("reasoning_content", messages[0])

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_empty_tool_calls_returns_messages_unchanged(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        messages = [{"role": "user", "content": "hi"}]
        result = handler.handle_function_calls([], messages)
        self.assertIs(result, messages)
        self.assertEqual(len(messages), 1)

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_missing_id_raises(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        with self.assertRaises(ValueError):
            handler.handle_function_calls(
                [{"function": {"name": "f", "arguments": "{}"}}], []
            )

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_missing_function_name_raises(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        with self.assertRaises(ValueError):
            handler.handle_function_calls(
                [{"id": "call_x", "function": {"arguments": "{}"}}], []
            )


class TestAskModelImageInput(unittest.TestCase):
    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_input_images_attached_to_last_user_message(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.id = "resp_img"
        handler.client.chat.completions.create.return_value = mock_response

        handler.ask_model(
            [{"role": "user", "content": "Describe this."}],
            input_images=["https://example.com/cat.png"],
        )

        sent = handler.client.chat.completions.create.call_args.kwargs["messages"]
        last_user = sent[-1]
        self.assertEqual(last_user["role"], "user")
        self.assertIsInstance(last_user["content"], list)
        parts = last_user["content"]
        self.assertTrue(any(p.get("type") == "text" for p in parts))
        self.assertTrue(any(p.get("type") == "image_url" for p in parts))
        image_part = next(p for p in parts if p.get("type") == "image_url")
        self.assertEqual(image_part["image_url"]["url"], "https://example.com/cat.png")

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_input_images_synthesizes_user_when_none_present(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.id = "resp_img2"
        handler.client.chat.completions.create.return_value = mock_response

        handler.ask_model(
            [{"role": "assistant", "content": "previous reply"}],
            input_images=["data:image/png;base64,iVBORw0KGgo="],
        )

        sent = handler.client.chat.completions.create.call_args.kwargs["messages"]
        # last message should be a user-role image-only message
        self.assertEqual(sent[-1]["role"], "user")
        self.assertEqual(sent[-1]["content"][0]["type"], "image_url")

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_input_images_rejects_bad_scheme(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.client = MagicMock()
        with self.assertRaises(Exception):
            handler.ask_model(
                [{"role": "user", "content": "?"}],
                input_images=["ftp://example.com/x.png"],
            )

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_input_images_with_user_role_instructions_gemma(self, mock_openai):
        """Gemma path: instructions_role=user must prepend instructions to the
        same list-content user message that _attach_images built."""
        logger, agent, setting = _make_handler(
            config_overrides={"instructions_role": "user"},
            agent_overrides={"instructions": "Be concise."},
        )
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.id = "resp_gemma"
        handler.client.chat.completions.create.return_value = mock_response

        handler.ask_model(
            [{"role": "user", "content": "Describe."}],
            input_images=["https://example.com/cat.png"],
        )

        sent = handler.client.chat.completions.create.call_args.kwargs["messages"]
        # Exactly one user message, no separate system/developer message.
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["role"], "user")
        parts = sent[0]["content"]
        # Order: instructions text -> original user text -> image_url
        self.assertEqual(parts[0], {"type": "text", "text": "Be concise."})
        self.assertEqual(parts[1], {"type": "text", "text": "Describe."})
        self.assertEqual(parts[2]["type"], "image_url")
        self.assertEqual(parts[2]["image_url"]["url"], "https://example.com/cat.png")


class TestToolShapeNormalization(unittest.TestCase):
    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_flat_responses_shape_converted_to_nested(self, mock_openai):
        # Simulate what mcp_http_client.export_tools_for_llm("gpt", ...) produces.
        flat_tool = {
            "type": "function",
            "name": "search_docs",
            "description": "Search the docs.",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        }
        logger, agent, setting = _make_handler(config_overrides={"tools": [flat_tool]})
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        tools = handler.model_setting.get("tools")
        self.assertIsNotNone(tools)
        self.assertEqual(len(tools), 1)
        t = tools[0]
        self.assertEqual(t["type"], "function")
        self.assertIn("function", t)
        self.assertEqual(t["function"]["name"], "search_docs")
        self.assertEqual(t["function"]["description"], "Search the docs.")
        self.assertEqual(
            t["function"]["parameters"]["properties"]["q"]["type"], "string"
        )

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_already_nested_shape_passes_through(self, mock_openai):
        nested_tool = {
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": "Search.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        logger, agent, setting = _make_handler(config_overrides={"tools": [nested_tool]})
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        self.assertEqual(handler.model_setting["tools"][0], nested_tool)

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_mixed_shapes_normalized_independently(self, mock_openai):
        flat = {"type": "function", "name": "flat_one", "parameters": {}}
        nested = {
            "type": "function",
            "function": {"name": "nested_one", "parameters": {}},
        }
        logger, agent, setting = _make_handler(
            config_overrides={"tools": [flat, nested]}
        )
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        tools = handler.model_setting["tools"]
        self.assertEqual(tools[0]["function"]["name"], "flat_one")
        self.assertEqual(tools[1]["function"]["name"], "nested_one")


class TestStructuredLogging(unittest.TestCase):
    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_model_call_log_emits_expected_fields(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)

        with self.assertLogs("test", level="INFO") as cm:
            handler._log_model_call(
                model="gpt-4o",
                request_id="req_1",
                stream=False,
                tool_call_count=2,
                usage=usage,
                latency_ms=123.4,
                finish_reason="stop",
                retry_count=0,
            )

        out = "\n".join(cm.output)
        self.assertIn("[MODEL_CALL]", out)
        self.assertIn("model=gpt-4o", out)
        self.assertIn("request_id=req_1", out)
        self.assertIn("stream=False", out)
        self.assertIn("tool_calls=2", out)
        self.assertIn("prompt_tokens=10", out)
        self.assertIn("completion_tokens=20", out)
        self.assertIn("total_tokens=30", out)
        self.assertIn("latency_ms=123.4", out)
        self.assertIn("finish_reason=stop", out)
        self.assertIn("retry_count=0", out)

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_tool_call_log_emits_call_and_result(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.get_function = MagicMock(return_value=lambda city: {"city": city, "t": 18})
        handler.invoke_async_funct = MagicMock()

        with self.assertLogs("test", level="INFO") as cm:
            handler._execute_function(
                {"id": "call_z", "name": "get_weather", "arguments": "{}", "type": "function"},
                {"city": "Tokyo"},
            )

        out = "\n".join(cm.output)
        self.assertIn("[TOOL_CALL]", out)
        self.assertIn("name=get_weather", out)
        self.assertIn("id=call_z", out)
        self.assertIn("[TOOL_RESULT]", out)
        self.assertIn("status=completed", out)

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_mcp_text_content_tool_output_is_unwrapped(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.get_function = MagicMock(
            return_value=lambda **kwargs: [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "request_uuid": "req_123",
                            "status": "initial",
                            "email": "buyer@example.com",
                        }
                    ),
                }
            ]
        )
        handler.invoke_async_funct = MagicMock()

        function_output, serialized_output = handler._execute_function(
            {
                "id": "call_rfq",
                "name": "submit_rfq_request",
                "arguments": "{}",
                "type": "function",
            },
            {},
        )

        self.assertEqual(function_output["request_uuid"], "req_123")
        self.assertNotIn('"type": "text"', serialized_output)
        self.assertEqual(json.loads(serialized_output)["status"], "initial")

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_truncate_caps_long_strings(self, mock_openai):
        from openai_completions_agent_handler.openai_completions_agent_handler import (
            _truncate,
        )

        self.assertEqual(_truncate("short"), "short")
        long_s = "x" * 500
        truncated = _truncate(long_s, limit=200)
        self.assertTrue(truncated.startswith("x" * 200))
        self.assertIn("(500 chars)", truncated)


class TestThinkTagSplitter(unittest.TestCase):
    def test_passthrough_no_tags(self):
        s = _ThinkTagSplitter()
        c, t = s.feed("hello world")
        # No tags ever appear; "hello world" is short so we keep last 6 chars buffered
        # in case a partial <think> tag is forming. Flush emits the rest.
        c2, t2 = s.flush()
        self.assertEqual(c + c2, "hello world")
        self.assertEqual(t + t2, "")

    def test_complete_block_in_one_feed(self):
        s = _ThinkTagSplitter()
        c, t = s.feed("<think>reason here</think>final answer")
        # After feed, leading content was empty, all of the reason went to think,
        # then "final answer" is buffered (could be partial tag at tail).
        c2, t2 = s.flush()
        self.assertEqual(t + t2, "reason here")
        self.assertEqual(c + c2, "final answer")

    def test_block_split_across_chunks(self):
        s = _ThinkTagSplitter()
        chunks = ["before <thi", "nk>reaso", "n</thi", "nk>after"]
        out_content = ""
        out_think = ""
        for ch in chunks:
            c, t = s.feed(ch)
            out_content += c
            out_think += t
        c, t = s.flush()
        out_content += c
        out_think += t
        self.assertEqual(out_content, "before after")
        self.assertEqual(out_think, "reason")

    def test_open_tag_at_chunk_boundary(self):
        s = _ThinkTagSplitter()
        # The < might be the start of <think>, so it should be held until we
        # see more.
        c, t = s.feed("abc<")
        self.assertEqual(c, "abc")
        self.assertEqual(t, "")
        c, t = s.feed("think>")
        self.assertEqual(c, "")
        c, t = s.feed("X</think>Y")
        c2, t2 = s.flush()
        self.assertEqual(t + t2, "X")
        self.assertEqual(c + c2, "Y")

    def test_unterminated_think_block_flushes_as_thinking(self):
        s = _ThinkTagSplitter()
        c1, t1 = s.feed("<think>still thinking when stream ends")
        c2, t2 = s.flush()
        self.assertEqual(c1 + c2, "")
        self.assertEqual(t1 + t2, "still thinking when stream ends")


class TestToolCallArgumentsCoercion(unittest.TestCase):
    """Verify tool_call arguments are kept as JSON strings (spec-compliant)."""

    def test_coerce_preserves_valid_json_string(self):
        raw = '{"key": "value", "n": 3}'
        result = OpenAICompletionsEventHandler._coerce_tool_call_arguments(raw)
        self.assertEqual(result, raw)
        self.assertIsInstance(result, str)

    def test_coerce_returns_dict_unchanged(self):
        d = {"already": "dict"}
        self.assertIs(
            OpenAICompletionsEventHandler._coerce_tool_call_arguments(d), d
        )

    def test_coerce_returns_non_json_string_unchanged(self):
        raw = "not json at all"
        self.assertEqual(
            OpenAICompletionsEventHandler._coerce_tool_call_arguments(raw), raw
        )

    def test_coerce_preserves_empty_object_string(self):
        self.assertEqual(
            OpenAICompletionsEventHandler._coerce_tool_call_arguments("{}"),
            "{}",
        )

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_append_assistant_with_tool_calls_keeps_string_arguments(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        messages = [{"role": "user", "content": "Hi"}]
        handler._append_assistant_with_tool_calls(
            [
                {
                    "id": "call_1",
                    "name": "submit_rfq_request",
                    "arguments": '{"item": "widget", "qty": 2}',
                    "type": "function",
                }
            ],
            messages,
        )
        args = messages[1]["tool_calls"][0]["function"]["arguments"]
        # Arguments stay as a JSON string per the OpenAI spec.
        self.assertIsInstance(args, str)
        self.assertEqual(args, '{"item": "widget", "qty": 2}')

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_invoke_model_stringifies_tool_call_arguments(self, mock_openai):
        logger, agent, setting = _make_handler()
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.client = MagicMock()
        handler.client.chat.completions.create.return_value = MagicMock()
        messages = [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "NYC"},
                        },
                    }
                ],
            },
        ]
        handler.invoke_model(messages=messages, stream=False)
        # invoke_model stringifies tool_call arguments in-place before the API call.
        tc_args = messages[1]["tool_calls"][0]["function"]["arguments"]
        self.assertIsInstance(tc_args, str)
        self.assertEqual(json.loads(tc_args), {"city": "NYC"})


class TestValidateToolMessages(unittest.TestCase):
    """Verify the pre-flight validator catches malformed `role: "tool"` messages."""

    @staticmethod
    def _ok(*args, **kwargs):
        return None

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_valid_messages_pass(self, mock_openai):
        OpenAICompletionsEventHandler._validate_tool_messages(
            [
                {"role": "system", "content": "instructions"},
                {"role": "user", "content": "Hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": {}}}
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "{}"},
            ]
        )


    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_missing_tool_call_id_raises(self, mock_openai):
        with self.assertRaises(ValueError) as ctx:
            OpenAICompletionsEventHandler._validate_tool_messages(
                [
                    {"role": "user", "content": "Hi"},
                    {"role": "tool", "content": "{}"},  # missing tool_call_id
                ]
            )
        self.assertIn("missing required 'tool_call_id'", str(ctx.exception))
        self.assertIn("messages[1]", str(ctx.exception))

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_orphan_tool_id_raises(self, mock_openai):
        # tool_call_id present but no matching assistant tool_call
        with self.assertRaises(ValueError) as ctx:
            OpenAICompletionsEventHandler._validate_tool_messages(
                [
                    {"role": "user", "content": "Hi"},
                    {"role": "tool", "tool_call_id": "orphan_id", "content": "{}"},
                ]
            )
        self.assertIn("no preceding assistant message", str(ctx.exception))

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_empty_messages_pass(self, mock_openai):
        OpenAICompletionsEventHandler._validate_tool_messages([])

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_non_tool_messages_pass(self, mock_openai):
        OpenAICompletionsEventHandler._validate_tool_messages(
            [
                {"role": "system", "content": "x"},
                {"role": "user", "content": "y"},
                {"role": "assistant", "content": "z"},
            ]
        )

class TestToolHistoryCompatibility(unittest.TestCase):
    @staticmethod
    def _messages():
        return [
            {"role": "user", "content": "Create RFQ"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_rfq",
                        "type": "function",
                        "function": {
                            "name": "submit_rfq_request",
                            "arguments": {"email": "buyer@example.com"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_rfq",
                "content": json.dumps({"request_uuid": "req_123"}),
            },
        ]

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_auto_flattens_together_glm_tool_history(self, mock_openai):
        logger, agent, setting = _make_handler(
            config_overrides={
                "base_url": "https://api.together.ai/v1",
                "model": "zai-org/GLM-5.2",
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "submit_rfq_request", "parameters": {}},
                    }
                ],
            }
        )
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.client = MagicMock()
        handler.client.chat.completions.create.return_value = MagicMock()

        handler.invoke_model(messages=self._messages(), stream=False)

        sent_messages = handler.client.chat.completions.create.call_args.kwargs["messages"]
        self.assertFalse(any(msg.get("role") == "tool" for msg in sent_messages))
        self.assertFalse(any("tool_calls" in msg for msg in sent_messages))
        self.assertIn("Tool result from submit_rfq_request", sent_messages[-1]["content"])

    @patch("openai_completions_agent_handler.openai_completions_agent_handler.openai.OpenAI")
    def test_auto_keeps_non_together_tool_history_strict(self, mock_openai):
        logger, agent, setting = _make_handler(
            config_overrides={
                "base_url": "http://127.0.0.1:11434/v1",
                "model": "qwen3",
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "submit_rfq_request", "parameters": {}},
                    }
                ],
            }
        )
        handler = OpenAICompletionsEventHandler(logger, agent, **setting)
        handler.client = MagicMock()
        handler.client.chat.completions.create.return_value = MagicMock()
        messages = self._messages()

        handler.invoke_model(messages=messages, stream=False)

        sent_messages = handler.client.chat.completions.create.call_args.kwargs["messages"]
        self.assertTrue(any(msg.get("role") == "tool" for msg in sent_messages))
        self.assertTrue(any("tool_calls" in msg for msg in sent_messages))


if __name__ == "__main__":
    unittest.main()
