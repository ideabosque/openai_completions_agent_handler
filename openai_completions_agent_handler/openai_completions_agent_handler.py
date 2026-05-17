#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

__author__ = "bibow"

import json
import logging
import re
import threading
import traceback
from decimal import Decimal
from queue import Queue
from typing import Any, Dict, List, Optional, Union

import httpx
import openai
import pendulum

from ai_agent_handler import AIAgentEventHandler
from silvaengine_utility import Debugger, Serializer
from silvaengine_utility.performance_monitor import performance_monitor


class ToolCallDepthExceeded(Exception):
    def __init__(self, depth: int, max_depth: int):
        self.depth = depth
        self.max_depth = max_depth
        super().__init__(
            f"Tool call recursion depth {depth} exceeds maximum {max_depth}"
        )


def _omit_none(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def _truncate(s: str, limit: int = 200) -> str:
    if s is None:
        return ""
    return s if len(s) <= limit else f"{s[:limit]}...({len(s)} chars)"


class OpenAICompletionsEventHandler(AIAgentEventHandler):
    """
    Manages conversations and function calls via OpenAI's Chat Completions API.
    Compatible with OpenAI, SGLang, vLLM, and other OpenAI-compatible servers.
    """

    def __init__(
        self,
        logger: logging.Logger,
        agent: Dict[str, Any],
        **setting: Dict[str, Any],
    ) -> None:
        try:
            AIAgentEventHandler.__init__(self, logger, agent, **setting)

            config = self.agent.get("configuration", {})
            request_timeout = float(config.get("request_timeout_seconds", 120.0))
            connect_timeout = float(config.get("connect_timeout_seconds", 10.0))
            pool_max = int(config.get("pool_max_connections", 20))
            pool_keepalive = int(config.get("pool_max_keepalive_connections", 10))

            self._http_client = httpx.Client(
                timeout=httpx.Timeout(timeout=request_timeout, connect=connect_timeout),
                limits=httpx.Limits(
                    max_connections=pool_max,
                    max_keepalive_connections=pool_keepalive,
                ),
            )

            sdk_max_retries = int(config.get("max_retries", 2))
            self.client = openai.OpenAI(
                api_key=config.get("openai_api_key"),
                base_url=config.get("base_url"),
                http_client=self._http_client,
                max_retries=sdk_max_retries,
            )

            if "enabled_tools" in config:
                enabled_set = set(config["enabled_tools"])
                if "tools" in config:
                    config["tools"] = [
                        tool
                        for tool in config["tools"]
                        if tool.get("function", {}).get("name") in enabled_set
                    ]

            self._assemble_extra_body(config)

            self.model_setting = {}
            for k, v in config.items():
                if k in [
                    "openai_api_key",
                    "base_url",
                    "enabled_tools",
                    "request_timeout_seconds",
                    "connect_timeout_seconds",
                    "pool_max_connections",
                    "pool_max_keepalive_connections",
                    "max_retries",
                    "max_tool_call_depth",
                    "instructions_role",
                    "enable_thinking",
                    "separate_reasoning",
                ]:
                    continue
                if k == "max_tokens":
                    v = int(v)
                elif k == "max_completion_tokens":
                    v = int(v)
                elif k in [
                    "temperature",
                    "top_p",
                    "presence_penalty",
                    "frequency_penalty",
                ]:
                    v = float(v)
                elif isinstance(v, Decimal):
                    v = float(v)
                self.model_setting[k] = v

            if (
                "max_completion_tokens" in self.model_setting
                and "max_tokens" in self.model_setting
            ):
                if self.logger and self.logger.isEnabledFor(logging.WARNING):
                    self.logger.warning(
                        "Both max_tokens and max_completion_tokens are set. "
                        "Preferring max_completion_tokens."
                    )
                del self.model_setting["max_tokens"]

            self._tools_list = self.model_setting.get("tools", [])
            if not self._tools_list:
                self.model_setting.pop("tools", None)
                self.model_setting.pop("tool_choice", None)
                self.model_setting.pop("parallel_tool_calls", None)

            self.instructions_role = str(config.get("instructions_role", "system"))
            self._max_tool_call_depth = int(config.get("max_tool_call_depth", 8))
            self.output_format_type = self.model_setting.get("response_format", {}).get(
                "type", "text"
            )
            self.enable_timeline_log = setting.get("enable_timeline_log", False)
            self._tools_cache_valid = False

            self._last_usage = None
            self._last_finish_reason = None
            self._last_latency_ms = None
            self._last_request_id = None
        except Exception as e:
            Debugger.info(variable=e, stage=f"{__name__}:__init__")
            raise

    def close(self) -> None:
        if hasattr(self, "_http_client") and self._http_client is not None:
            self._http_client.close()
            self._http_client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _check_retry_limit(self, retry_count: int, max_retries: int = 2) -> None:
        if retry_count > max_retries:
            error_msg = f"Maximum empty-response retry limit ({max_retries}) exceeded"
            if self.logger and self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(error_msg)
            raise Exception(error_msg)

    def _has_valid_content(self, text: Optional[str]) -> bool:
        return bool(text and text.strip())

    def _get_elapsed_time(self) -> float:
        if not hasattr(self, "_global_start_time") or self._global_start_time is None:
            return 0.0
        return (pendulum.now("UTC") - self._global_start_time).total_seconds() * 1000

    def reset_timeline(self) -> None:
        self._global_start_time = None
        if (
            self.enable_timeline_log
            and self.logger
            and self.logger.isEnabledFor(logging.INFO)
        ):
            self.logger.info("[TIMELINE] Timeline reset for new run")

    def _redact_api_key(self, text: str) -> str:
        return re.sub(r"(sk-|key[\"':\s]*)\S{4,}", r"\1****", text)

    def _log_model_call(
        self,
        model: str,
        request_id: Optional[str],
        stream: bool,
        tool_call_count: int,
        usage: Any,
        latency_ms: Optional[float],
        finish_reason: Optional[str],
        retry_count: int,
    ) -> None:
        if not self.logger or not self.logger.isEnabledFor(logging.INFO):
            return
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        total_tokens = getattr(usage, "total_tokens", None) if usage else None
        latency_str = f"{latency_ms:.1f}" if latency_ms is not None else "None"
        self.logger.info(
            f"[MODEL_CALL] model={model} request_id={request_id} stream={stream} "
            f"tool_calls={tool_call_count} prompt_tokens={prompt_tokens} "
            f"completion_tokens={completion_tokens} total_tokens={total_tokens} "
            f"latency_ms={latency_str} finish_reason={finish_reason} "
            f"retry_count={retry_count}"
        )

    def _assemble_extra_body(self, config: Dict[str, Any]) -> None:
        """
        Convert SGLang/Qwen3 shorthand config keys into the nested `extra_body`
        structure that chat.completions.create() expects.

        Recognized shorthand keys (top-level in `configuration`):
            enable_thinking    -> extra_body["chat_template_kwargs"]["enable_thinking"]
            separate_reasoning -> extra_body["separate_reasoning"]

        Anything the caller has already placed in `extra_body` wins; this method
        only fills missing slots so users can always override by setting
        `extra_body` directly.
        """
        shorthand: Dict[str, Any] = {}
        if "enable_thinking" in config and config["enable_thinking"] is not None:
            shorthand["chat_template_kwargs"] = {
                "enable_thinking": bool(config["enable_thinking"])
            }
        if "separate_reasoning" in config and config["separate_reasoning"] is not None:
            shorthand["separate_reasoning"] = bool(config["separate_reasoning"])

        if not shorthand:
            return

        explicit = config.get("extra_body") or {}
        merged: Dict[str, Any] = dict(shorthand)
        for k, v in explicit.items():
            if (
                k == "chat_template_kwargs"
                and isinstance(v, dict)
                and isinstance(merged.get(k), dict)
            ):
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v
        config["extra_body"] = merged

    def _merge_instructions_into_first_user(
        self, messages: List[Dict[str, Any]], instructions: str
    ) -> List[Dict[str, Any]]:
        merged = list(messages)
        for i, msg in enumerate(merged):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                merged[i] = {**msg, "content": f"{instructions}\n\n{content}"}
            elif isinstance(content, list):
                merged[i] = {
                    **msg,
                    "content": [{"type": "text", "text": instructions}] + content,
                }
            else:
                merged[i] = {**msg, "content": instructions}
            return merged
        return [{"role": "user", "content": instructions}] + merged

    def invoke_model(self, **kwargs: Dict[str, Any]) -> Any:
        try:
            if self.enable_timeline_log:
                invoke_start = pendulum.now("UTC")

            payload = dict(self.model_setting, **kwargs)

            messages = payload.get("messages") or []
            instructions = self.agent.get("instructions")
            if instructions and not (
                messages and messages[0].get("role") in {"system", "developer"}
            ):
                if self.instructions_role == "user":
                    messages = self._merge_instructions_into_first_user(
                        messages, instructions
                    )
                else:
                    messages = [
                        {"role": self.instructions_role, "content": instructions}
                    ] + messages
            payload["messages"] = messages

            if payload.get("stream"):
                payload["stream_options"] = {"include_usage": True}
            else:
                payload.pop("stream_options", None)

            result = self.client.chat.completions.create(**_omit_none(payload))

            if (
                self.enable_timeline_log
                and self.logger
                and self.logger.isEnabledFor(logging.INFO)
            ):
                invoke_time = (
                    pendulum.now("UTC") - invoke_start
                ).total_seconds() * 1000
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: API call returned (took {invoke_time:.2f}ms)"
                )

            return result
        except openai.BadRequestError as e:
            safe_msg = self._redact_api_key(str(e))
            if self.logger and self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(
                    f"BadRequestError: {safe_msg} "
                    f"(model={self.model_setting.get('model')}, "
                    f"has_tools={bool(self.model_setting.get('tools'))}, "
                    f"stream={bool(kwargs.get('stream'))})"
                )
            raise
        except Exception as e:
            safe_msg = self._redact_api_key(str(e))
            if self.logger and self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(f"Error invoking model: {safe_msg}")
            raise Exception(f"Failed to invoke model: {safe_msg}")

    @performance_monitor.monitor_operation(operation_name="OpenAICompletions")
    def ask_model(
        self,
        input_messages: List[Dict[str, Any]],
        queue: Queue = None,
        stream_event: threading.Event = None,
        model_setting: Dict[str, Any] = None,
    ) -> Optional[str]:
        if self.enable_timeline_log:
            ask_model_start = pendulum.now("UTC")

        if not hasattr(self, "_ask_model_depth"):
            self._ask_model_depth = 0
        self._ask_model_depth += 1
        is_top_level = self._ask_model_depth == 1

        if self._ask_model_depth > self._max_tool_call_depth:
            self._ask_model_depth -= 1
            raise ToolCallDepthExceeded(
                self._ask_model_depth, self._max_tool_call_depth
            )

        if is_top_level and self.enable_timeline_log:
            self._global_start_time = ask_model_start
            if self.logger and self.logger.isEnabledFor(logging.INFO):
                self.logger.info("[TIMELINE] T+0ms: Run started")
        else:
            if (
                self.enable_timeline_log
                and self.logger
                and self.logger.isEnabledFor(logging.INFO)
            ):
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Recursive ask_model call started (depth={self._ask_model_depth})"
                )

        try:
            if not self.client:
                if self.logger and self.logger.isEnabledFor(logging.ERROR):
                    self.logger.error("No OpenAI client provided.")
                return None

            stream = True if queue is not None else False

            if model_setting:
                self.model_setting.update(model_setting)
                self._tools_cache_valid = False

            if (
                self.enable_timeline_log
                and self.logger
                and self.logger.isEnabledFor(logging.INFO)
            ):
                preparation_time = (
                    pendulum.now("UTC") - ask_model_start
                ).total_seconds() * 1000
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Preparation complete (took {preparation_time:.2f}ms)"
                )

            call_start = pendulum.now("UTC")
            response = self.invoke_model(
                messages=input_messages,
                stream=stream,
            )
            latency_ms = (pendulum.now("UTC") - call_start).total_seconds() * 1000
            self._last_latency_ms = latency_ms

            run_id = None
            if stream:
                run_id = self.handle_stream(
                    response,
                    input_messages,
                    queue=queue,
                    stream_event=stream_event,
                )
            else:
                run_id = self.handle_response(response, input_messages)

            return run_id
        except ToolCallDepthExceeded:
            raise
        except Exception as e:
            safe_msg = self._redact_api_key(str(e))
            if self.logger and self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(f"Error in ask_model: {safe_msg}")
            raise Exception(f"Failed to process model request: {safe_msg}")
        finally:
            self._ask_model_depth -= 1
            if self._ask_model_depth == 0:
                if (
                    self.enable_timeline_log
                    and self.logger
                    and self.logger.isEnabledFor(logging.INFO)
                ):
                    elapsed = self._get_elapsed_time()
                    self.logger.info(f"[TIMELINE] T+{elapsed:.2f}ms: Run complete")
                self._global_start_time = None

    def _trim_messages_for_recursion(
        self, input_messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        max_messages = self.agent.get("num_of_messages")
        if not max_messages or len(input_messages) <= max_messages:
            return input_messages

        head_is_instructions = input_messages and input_messages[0].get("role") in {
            "system",
            "developer",
        }
        instructions = input_messages[:1] if head_is_instructions else []
        body = input_messages[len(instructions) :]
        trimmed = body[-max_messages:]

        while trimmed and trimmed[0].get("role") == "tool":
            trimmed = trimmed[1:]

        return instructions + trimmed

    def handle_function_call(
        self,
        tool_call: Dict[str, Any],
        input_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if self.enable_timeline_log:
            function_call_start = pendulum.now("UTC")

        if not tool_call or "id" not in tool_call:
            raise ValueError("Invalid tool_call object")
        if not tool_call.get("function", {}).get("name"):
            raise ValueError("Tool call missing function name")

        try:
            function_call_data = {
                "id": tool_call["id"],
                "name": tool_call["function"]["name"],
                "arguments": tool_call["function"].get("arguments", "{}"),
                "type": "function",
            }

            arguments = self._process_function_arguments(function_call_data)

            function_output, serialized_output = self._execute_function(
                function_call_data, arguments
            )

            self._update_conversation_history(
                function_call_data, function_output, input_messages, serialized_output
            )

            if self._run is None:
                self._short_term_memory.append(
                    {
                        "message": {
                            "role": self.agent["tool_call_role"],
                            "content": Serializer.json_dumps(
                                {
                                    "tool": {
                                        "tool_call_id": tool_call["id"],
                                        "tool_type": "function",
                                        "name": tool_call["function"]["name"],
                                        "arguments": arguments,
                                    },
                                    "output": function_output,
                                }
                            ),
                        },
                        "created_at": pendulum.now("UTC"),
                    }
                )

            if (
                self.enable_timeline_log
                and self.logger
                and self.logger.isEnabledFor(logging.INFO)
            ):
                function_call_time = (
                    pendulum.now("UTC") - function_call_start
                ).total_seconds() * 1000
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Function '{function_call_data['name']}' complete (took {function_call_time:.2f}ms)"
                )

            return input_messages

        except Exception as e:
            if self.logger and self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(f"Error in handle_function_call: {e}")
            raise

    def _process_function_arguments(
        self, function_call_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            arguments = Serializer.json_loads(function_call_data.get("arguments", "{}"))
            return arguments
        except Exception as e:
            log = traceback.format_exc()
            self.invoke_async_funct(
                module_name="ai_agent_core_engine",
                class_name="AIAgentCoreEngine",
                function_name="async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "arguments": function_call_data.get("arguments", "{}"),
                    "status": "failed",
                    "notes": log,
                },
            )
            if self.logger and self.logger.isEnabledFor(logging.ERROR):
                self.logger.error("Error parsing function arguments: %s", e)
            raise ValueError(f"Failed to parse function arguments: {e}")

    def _execute_function(
        self, function_call_data: Dict[str, Any], arguments: Dict[str, Any]
    ) -> Any:
        name = function_call_data["name"]
        tool_call_id = function_call_data["id"]

        agent_function = self.get_function(name)
        if not agent_function:
            if self.logger and self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(
                    f"[TOOL_CALL] id={tool_call_id} name={name} status=unknown_tool"
                )
            raise ValueError(f"Unsupported function requested: {name}")

        arguments_json = Serializer.json_dumps(arguments)

        if self.logger and self.logger.isEnabledFor(logging.INFO):
            self.logger.info(
                f"[TOOL_CALL] id={tool_call_id} name={name} "
                f"args={_truncate(arguments_json)}"
            )

        try:
            self.invoke_async_funct(
                module_name="ai_agent_core_engine",
                class_name="AIAgentCoreEngine",
                function_name="async_insert_update_tool_call",
                **{
                    "tool_call_id": tool_call_id,
                    "tool_type": function_call_data["type"],
                    "name": name,
                    "arguments": arguments_json,
                    "status": "in_progress",
                },
            )

            exec_start = pendulum.now("UTC")
            function_output = agent_function(**arguments)
            exec_ms = (pendulum.now("UTC") - exec_start).total_seconds() * 1000

            serialized_output = Serializer.json_dumps(function_output)

            if self.logger and self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"[TOOL_RESULT] id={tool_call_id} name={name} status=completed "
                    f"exec_ms={exec_ms:.1f} output_chars={len(serialized_output)} "
                    f"output={_truncate(serialized_output)}"
                )

            if (
                self.enable_timeline_log
                and self.logger
                and self.logger.isEnabledFor(logging.INFO)
            ):
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Function '{name}' executed (took {exec_ms:.2f}ms)"
                )

            self.invoke_async_funct(
                module_name="ai_agent_core_engine",
                class_name="AIAgentCoreEngine",
                function_name="async_insert_update_tool_call",
                **{
                    "tool_call_id": tool_call_id,
                    "content": serialized_output,
                    "status": "completed",
                },
            )
            return function_output, serialized_output

        except Exception as e:
            log = traceback.format_exc()
            if self.logger and self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(
                    f"[TOOL_RESULT] id={tool_call_id} name={name} status=failed "
                    f"error={e!r}"
                )
            self.invoke_async_funct(
                module_name="ai_agent_core_engine",
                class_name="AIAgentCoreEngine",
                function_name="async_insert_update_tool_call",
                **{
                    "tool_call_id": tool_call_id,
                    "arguments": arguments_json,
                    "status": "failed",
                    "notes": log,
                },
            )
            error_msg = f"Function execution failed: {e}"
            return error_msg, Serializer.json_dumps(error_msg)

    def _update_conversation_history(
        self,
        function_call_data: Dict[str, Any],
        function_output: Any,
        input_messages: List[Dict[str, Any]],
        serialized_output: Optional[str] = None,
    ) -> None:
        input_messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": function_call_data["id"],
                        "type": "function",
                        "function": {
                            "name": function_call_data["name"],
                            "arguments": function_call_data.get("arguments", "{}"),
                        },
                    }
                ],
            }
        )
        input_messages.append(
            {
                "role": "tool",
                "tool_call_id": function_call_data["id"],
                "content": (
                    serialized_output
                    if serialized_output is not None
                    else Serializer.json_dumps(function_output)
                ),
            }
        )

    def handle_response(
        self,
        response: Any,
        input_messages: List[Dict[str, Any]],
        retry_count: int = 0,
    ) -> str:
        self._check_retry_limit(retry_count)

        message = response.choices[0].message
        content = message.content or ""
        tool_calls = message.tool_calls
        reasoning_content = getattr(message, "reasoning_content", None)
        finish_reason = response.choices[0].finish_reason

        self._last_usage = response.usage
        self._last_finish_reason = finish_reason
        self._last_request_id = response.id

        tool_call_count = len(tool_calls) if tool_calls else 0
        self._log_model_call(
            model=self.model_setting.get("model", ""),
            request_id=response.id,
            stream=False,
            tool_call_count=tool_call_count,
            usage=response.usage,
            latency_ms=self._last_latency_ms,
            finish_reason=finish_reason,
            retry_count=retry_count,
        )

        if reasoning_content:
            self.final_output["reasoning_summary"] = reasoning_content

        if tool_calls:
            for tool_call in tool_calls:
                input_messages = self.handle_function_call(
                    {
                        "id": tool_call.id,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    },
                    input_messages,
                )
            input_messages = self._trim_messages_for_recursion(input_messages)
            self.ask_model(input_messages)
            return response.id

        if finish_reason == "length":
            self.final_output["truncated"] = True
        if finish_reason == "content_filter":
            self.final_output["filtered"] = True

        if not self._has_valid_content(content):
            if self.logger and self.logger.isEnabledFor(logging.WARNING):
                self.logger.warning(
                    f"Empty response, retrying (attempt {retry_count + 1}/2)..."
                )
            next_response = self.invoke_model(
                messages=input_messages,
                stream=False,
            )
            self.handle_response(
                next_response, input_messages, retry_count=retry_count + 1
            )
            return response.id

        self.final_output.update(
            {
                "message_id": response.id,
                "role": message.role,
                "content": content,
            }
        )

        return response.id

    def handle_stream(
        self,
        response_stream: Any,
        input_messages: List[Dict[str, Any]],
        queue: Queue = None,
        stream_event: threading.Event = None,
        retry_count: int = 0,
    ) -> str | None:
        self._check_retry_limit(retry_count)

        run_id = None
        role = None
        accumulated_text_parts = []
        accumulated_partial_text_parts = []
        accumulated_partial_json_parts = []
        accumulated_reasoning_parts = []
        received_any_content = False
        output_format = self.output_format_type
        index = 0
        tool_call_count = 0
        finish_reason = None

        tool_call_accumulators: Dict[int, Dict[str, Any]] = {}

        if self.enable_timeline_log:
            stream_start_time = pendulum.now("UTC")

        for chunk in response_stream:
            if run_id is None:
                run_id = chunk.id
                if queue:
                    queue.put({"name": "run_id", "value": chunk.id})

            if not chunk.choices:
                if chunk.usage:
                    self._last_usage = chunk.usage
                continue

            choice = chunk.choices[0]
            delta = choice.delta
            finish_reason = choice.finish_reason

            if getattr(delta, "content", None):
                received_any_content = True
                accumulated_text_parts.append(delta.content)

                if output_format in ["json_object", "json_schema"]:
                    accumulated_partial_json_parts.append(delta.content)
                    temp_text = "".join(accumulated_text_parts)
                    index, temp_text, remaining_json = self.process_and_send_json(
                        index,
                        temp_text,
                        "".join(accumulated_partial_json_parts),
                        output_format,
                    )
                    accumulated_partial_json_parts = (
                        [remaining_json] if remaining_json else []
                    )
                else:
                    accumulated_partial_text_parts.append(delta.content)
                    index, remaining_text = self.process_text_content(
                        index, "".join(accumulated_partial_text_parts), output_format
                    )
                    accumulated_partial_text_parts = (
                        [remaining_text] if remaining_text else []
                    )

            reasoning_content = getattr(delta, "reasoning_content", None)
            if reasoning_content and isinstance(reasoning_content, str):
                accumulated_reasoning_parts.append(reasoning_content)

            if getattr(delta, "tool_calls", None):
                received_any_content = True
                tool_call_count += 1
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    acc = tool_call_accumulators.setdefault(
                        idx,
                        {"id": None, "type": "function", "name": "", "arguments": ""},
                    )
                    if getattr(tc_delta, "id", None):
                        acc["id"] = tc_delta.id
                    if getattr(tc_delta, "function", None):
                        if getattr(tc_delta.function, "name", None):
                            acc["name"] += tc_delta.function.name
                        if getattr(tc_delta.function, "arguments", None):
                            acc["arguments"] += tc_delta.function.arguments

            if finish_reason:
                self._last_finish_reason = finish_reason

                if finish_reason == "tool_calls":
                    tool_calls = [
                        {
                            "id": acc["id"],
                            "function": {
                                "name": acc["name"],
                                "arguments": acc["arguments"],
                            },
                        }
                        for acc in tool_call_accumulators.values()
                    ]
                    for tool_call in tool_calls:
                        input_messages = self.handle_function_call(
                            tool_call, input_messages
                        )
                    input_messages = self._trim_messages_for_recursion(input_messages)
                    self.ask_model(
                        input_messages, queue=queue, stream_event=stream_event
                    )
                    return run_id

                if finish_reason == "length":
                    self.final_output["truncated"] = True
                    if self.logger and self.logger.isEnabledFor(logging.WARNING):
                        self.logger.warning(
                            "Stream finished with reason: length (truncated)"
                        )

                if finish_reason == "content_filter":
                    self.final_output["filtered"] = True
                    if self.logger and self.logger.isEnabledFor(logging.WARNING):
                        self.logger.warning(
                            "Stream finished with reason: content_filter"
                        )

                if finish_reason == "stop":
                    role = getattr(delta, "role", None) or "assistant"

                if finish_reason not in (
                    "stop",
                    "tool_calls",
                    "length",
                    "content_filter",
                ):
                    if self.logger and self.logger.isEnabledFor(logging.WARNING):
                        self.logger.warning(
                            f"Unknown finish_reason: {finish_reason}, treating as stop"
                        )
                    role = getattr(delta, "role", None) or "assistant"

        final_accumulated_text = "".join(accumulated_text_parts)

        self._last_request_id = run_id
        self._log_model_call(
            model=self.model_setting.get("model", ""),
            request_id=run_id,
            stream=True,
            tool_call_count=len(tool_call_accumulators),
            usage=self._last_usage,
            latency_ms=self._last_latency_ms,
            finish_reason=finish_reason,
            retry_count=retry_count,
        )

        if not received_any_content:
            if self.logger and self.logger.isEnabledFor(logging.WARNING):
                self.logger.warning(
                    f"Empty stream, retrying (attempt {retry_count + 1}/2)..."
                )
            next_response = self.invoke_model(
                messages=input_messages,
                stream=True,
            )
            self.handle_stream(
                next_response,
                input_messages,
                queue=queue,
                stream_event=stream_event,
                retry_count=retry_count + 1,
            )
            return run_id

        if accumulated_reasoning_parts:
            self.final_output["reasoning_summary"] = "".join(
                accumulated_reasoning_parts
            )

        self.final_output.update(
            {
                "message_id": run_id,
                "role": role or "assistant",
                "content": final_accumulated_text,
            }
        )
        self.accumulated_text = final_accumulated_text

        if stream_event:
            stream_event.set()

        return run_id
