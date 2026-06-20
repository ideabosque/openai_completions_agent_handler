#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

__author__ = "bibow"

import logging
import re
import threading
import traceback
from decimal import Decimal
from queue import Queue
from typing import Any, Dict, List, Optional

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


class _ThinkTagSplitter:
    """
    Stream-aware splitter for inline <think>...</think> reasoning blocks.

    Used when an upstream server returns reasoning bundled into `content`
    rather than a dedicated field. Each call to feed(chunk) returns
    (content, thinking) pieces ready to emit. Partial tag matches at the
    buffer tail are held until enough characters arrive to split them.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self.in_think = False
        self.buffer = ""

    @staticmethod
    def _partial_tag_suffix(text: str, tag: str) -> int:
        max_length = min(len(text), len(tag) - 1)
        for length in range(max_length, 0, -1):
            if text.endswith(tag[:length]):
                return length
        return 0

    def feed(self, chunk: str) -> tuple[str, str]:
        self.buffer += chunk
        content_parts: List[str] = []
        think_parts: List[str] = []
        while self.buffer:
            if not self.in_think:
                idx = self.buffer.find(self.OPEN)
                if idx == -1:
                    keep = self._partial_tag_suffix(self.buffer, self.OPEN)
                    if len(self.buffer) > keep:
                        content_parts.append(
                            self.buffer[:-keep] if keep else self.buffer
                        )
                        self.buffer = self.buffer[-keep:] if keep else ""
                    break
                if idx > 0:
                    content_parts.append(self.buffer[:idx])
                self.buffer = self.buffer[idx + len(self.OPEN) :]
                self.in_think = True
            else:
                idx = self.buffer.find(self.CLOSE)
                if idx == -1:
                    keep = self._partial_tag_suffix(self.buffer, self.CLOSE)
                    if len(self.buffer) > keep:
                        think_parts.append(self.buffer[:-keep] if keep else self.buffer)
                        self.buffer = self.buffer[-keep:] if keep else ""
                    break
                if idx > 0:
                    think_parts.append(self.buffer[:idx])
                self.buffer = self.buffer[idx + len(self.CLOSE) :]
                self.in_think = False
        return "".join(content_parts), "".join(think_parts)

    def flush(self) -> tuple[str, str]:
        tail = self.buffer
        self.buffer = ""
        if self.in_think:
            return "", tail
        return tail, ""


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

            # Normalize tool shape BEFORE enabled_tools filtering, since
            # mcp_http_client.export_tools_for_llm("gpt", ...) emits the
            # Responses API flat shape {"type":"function","name":...,
            # "parameters":...} while Chat Completions requires the nested
            # shape {"type":"function","function":{"name":..., ...}}.
            if "tools" in config and config["tools"]:
                config["tools"] = self._normalize_tools_to_chat_completions(
                    config["tools"]
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
                    "enable_think_tag_split",
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
            elif self.logger and self.logger.isEnabledFor(logging.INFO):
                # One-time diagnostic so on-wire tool shape is visible in Lambda logs.
                bad = [
                    t
                    for t in self._tools_list
                    if not (isinstance(t, dict) and isinstance(t.get("function"), dict))
                ]
                self.logger.info(
                    f"[TOOLS_LOADED] count={len(self._tools_list)} "
                    f"malformed={len(bad)} "
                    f"names={[(t.get('function') or {}).get('name') for t in self._tools_list]}"
                )
                if bad:
                    self.logger.warning(
                        f"[TOOLS_MALFORMED] {len(bad)} tool(s) missing nested "
                        f"`function` field will be rejected by Chat Completions. "
                        f"First bad tool keys: {list(bad[0].keys()) if isinstance(bad[0], dict) else type(bad[0])}"
                    )

            self.instructions_role = str(config.get("instructions_role", "system"))
            self._max_tool_call_depth = int(config.get("max_tool_call_depth", 8))
            self._enable_think_tag_split = bool(
                config.get("enable_think_tag_split", False)
            )
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

    @staticmethod
    def _get_reasoning_content(part: Any) -> Optional[str]:
        for field_name in ("reasoning_content", "reasoning", "thinking"):
            value = getattr(part, field_name, None)
            if isinstance(value, str) and value:
                return value
        return None

    @staticmethod
    def _split_inline_reasoning(content: str) -> tuple[str, str]:
        splitter = _ThinkTagSplitter()
        visible, reasoning = splitter.feed(content)
        tail_visible, tail_reasoning = splitter.flush()
        return visible + tail_visible, reasoning + tail_reasoning

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
        Convert SGLang/Qwen3 shorthand config keys into the nested
        `extra_body` structure that chat.completions.create() expects.

        Recognized shorthand keys (top-level in `configuration`):
            enable_thinking    -> extra_body["chat_template_kwargs"]["enable_thinking"]
            separate_reasoning -> extra_body["separate_reasoning"]

        Z.AI GLM expects `extra_body={"thinking": {"type": "enabled"}}`.
        Supply that explicitly rather than combining two providers' extension
        keys in the same request.

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

    @staticmethod
    def _normalize_tools_to_chat_completions(
        tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Normalize tool definitions to the Chat Completions nested shape.

        Accepted input shapes (preserved or converted to the same output):
            Responses API flat:
                {"type":"function","name":"x","description":"...","parameters":{...}}
            Chat Completions nested:
                {"type":"function","function":{"name":"x","description":"...","parameters":{...}}}

        Anything that doesn't match either pattern is passed through unchanged.
        """
        normalized: List[Dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                normalized.append(tool)
                continue
            if isinstance(tool.get("function"), dict):
                normalized.append(tool)
                continue
            if tool.get("type") == "function" and "name" in tool:
                normalized.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {}),
                        },
                    }
                )
                continue
            normalized.append(tool)
        return normalized

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

    @staticmethod
    def _validate_image_url(url: str) -> None:
        """Raise ValueError if the image URL scheme is unsupported."""
        allowed = {"http", "https", "data"}
        scheme = url.split(":", 1)[0].lower()
        if scheme not in allowed:
            raise ValueError(
                f"Unsupported image URL scheme '{scheme}'. "
                f"Allowed schemes: {', '.join(sorted(allowed))}."
            )

    @staticmethod
    def _build_image_message(
        image_url: str, text: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build a Chat Completions content array containing an image."""
        OpenAICompletionsEventHandler._validate_image_url(image_url)
        content: List[Dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
        if text:
            content.insert(0, {"type": "text", "text": text})
        return {"role": "user", "content": content}

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
        except openai.APIError as e:
            safe_msg = self._redact_api_key(str(e))
            if self.logger and self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(f"OpenAI APIError: {safe_msg}")
            raise
        except Exception as e:
            safe_msg = self._redact_api_key(str(e))
            if self.logger and self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(f"Error invoking model: {safe_msg}")
            raise Exception(f"Failed to invoke model: {safe_msg}")

    def _attach_images(
        self,
        input_messages: List[Dict[str, Any]],
        image_urls: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Attach image_url content parts to the last user message.

        If `input_messages` has no trailing user message, the first image gets
        its own user message via `_build_image_message`; remaining images are
        appended to it. The caller's list is not mutated.
        """
        if not image_urls:
            return input_messages

        input_messages = list(input_messages)

        if not input_messages or input_messages[-1].get("role") != "user":
            input_messages.append(self._build_image_message(image_urls[0]))
            image_urls = image_urls[1:]

        if not image_urls:
            return input_messages

        last = input_messages[-1]
        existing = last.get("content")
        if isinstance(existing, str):
            content = [{"type": "text", "text": existing}] if existing else []
        elif isinstance(existing, list):
            content = list(existing)
        else:
            content = []

        for url in image_urls:
            self._validate_image_url(url)
            content.append({"type": "image_url", "image_url": {"url": url}})

        input_messages[-1] = {**last, "content": content}
        return input_messages

    def _send_terminal_stream_error(
        self,
        message: str,
        stream_event: threading.Event = None,
    ) -> None:
        """Emit a final stream frame when ask_model fails mid-stream."""
        try:
            self.send_data_to_stream(
                index=0,
                data_format="text",
                chunk_delta=f"\n[error] {message}",
                is_message_end=True,
            )
        except Exception as exc:
            if self.logger and self.logger.isEnabledFor(logging.WARNING):
                self.logger.warning(
                    "Failed to emit terminal stream error frame: %s",
                    exc,
                )
        finally:
            if stream_event:
                stream_event.set()

    @performance_monitor.monitor_operation(operation_name="OpenAICompletions")
    def ask_model(
        self,
        input_messages: List[Dict[str, Any]],
        queue: Queue = None,
        stream_event: threading.Event = None,
        input_images: Optional[List[str]] = None,
        model_setting: Dict[str, Any] = None,
    ) -> Optional[str]:
        if self.enable_timeline_log:
            ask_model_start = pendulum.now("UTC")

        if not hasattr(self, "_ask_model_depth"):
            self._ask_model_depth = 0
        self._ask_model_depth += 1
        is_top_level = self._ask_model_depth == 1

        if self._ask_model_depth > self._max_tool_call_depth:
            attempted_depth = self._ask_model_depth
            exc = ToolCallDepthExceeded(attempted_depth, self._max_tool_call_depth)
            if is_top_level and queue is not None:
                self._send_terminal_stream_error(
                    self._redact_api_key(str(exc)),
                    stream_event,
                )
            self._ask_model_depth -= 1
            raise exc

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

            if input_images:
                input_messages = self._attach_images(input_messages, input_images)

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
        except ToolCallDepthExceeded as e:
            safe_msg = self._redact_api_key(str(e))
            if is_top_level and stream:
                self._send_terminal_stream_error(safe_msg, stream_event)
            raise
        except openai.APIError as e:
            safe_msg = self._redact_api_key(str(e))
            if is_top_level and stream:
                self._send_terminal_stream_error(safe_msg, stream_event)
            raise
        except Exception as e:
            safe_msg = self._redact_api_key(str(e))
            if self.logger and self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(f"Error in ask_model: {safe_msg}")
            if is_top_level and stream:
                self._send_terminal_stream_error(safe_msg, stream_event)
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
        # Only a leading system/developer message is preserved here.
        # For instructions_role="user" (Gemma family), instructions are NOT in
        # input_messages — they are merged into the first user message inside
        # invoke_model on every request — so there is nothing to preserve at the
        # head and the window can safely slide over all user/assistant/tool turns.
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

    def handle_function_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        input_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Execute a batch of tool_calls that came back from a single model turn.

        Chat Completions requires one assistant message containing ALL of the
        turn's tool_calls, followed by one `role:"tool"` result per call —
        emitting separate assistant messages per call is rejected by stricter
        provider templates.
        """
        if not tool_calls:
            return input_messages

        validated = []
        for tc in tool_calls:
            if not tc or "id" not in tc:
                raise ValueError("Invalid tool_call object")
            if not tc.get("function", {}).get("name"):
                raise ValueError("Tool call missing function name")
            validated.append(
                {
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": tc["function"].get("arguments", "{}"),
                    "type": "function",
                }
            )

        if self.logger and self.logger.isEnabledFor(logging.INFO):
            self.logger.info(
                f"[handle_function_calls] Processing {len(validated)} tool call(s): "
                f"{[fcd['name'] for fcd in validated]}"
            )

        self._append_assistant_with_tool_calls(validated, input_messages)

        for fcd in validated:
            self._execute_and_append_result(fcd, input_messages)

        return input_messages

    def _execute_and_append_result(
        self,
        function_call_data: Dict[str, Any],
        input_messages: List[Dict[str, Any]],
    ) -> None:
        name = function_call_data["name"]
        if self.enable_timeline_log:
            call_start = pendulum.now("UTC")

        try:
            if self.logger and self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"[handle_function_call] Processing arguments for function {name}"
                )
            arguments = self._process_function_arguments(function_call_data)

            if self.logger and self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"[handle_function_call] Executing function {name} with arguments "
                    f"{_truncate(Serializer.json_dumps(arguments))}"
                )
            function_output, serialized_output = self._execute_function(
                function_call_data, arguments
            )

            if self.logger and self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"[handle_function_call][{name}] Updating conversation history"
                )
            self._append_tool_result(
                function_call_data["id"], serialized_output, input_messages
            )

            if self._run is None:
                self._short_term_memory.append(
                    {
                        "message": {
                            "role": self.agent["tool_call_role"],
                            "content": Serializer.json_dumps(
                                {
                                    "tool": {
                                        "tool_call_id": function_call_data["id"],
                                        "tool_type": "function",
                                        "name": function_call_data["name"],
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
                call_ms = (pendulum.now("UTC") - call_start).total_seconds() * 1000
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Function '{function_call_data['name']}' complete (took {call_ms:.2f}ms)"
                )

        except Exception as e:
            if self.logger and self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(
                    f"Error executing tool {function_call_data['name']}: {e}"
                )
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

    def _append_assistant_with_tool_calls(
        self,
        function_call_data_list: List[Dict[str, Any]],
        input_messages: List[Dict[str, Any]],
    ) -> None:
        """
        Append ONE assistant message containing every tool_call from this turn.

        Note: `reasoning_content` is intentionally NOT included on input
        messages. It is a response-only field; echoing it back to the model
        on the next turn is non-standard and is rejected by strict providers
        such as Groq's openai/gpt-oss-* family and OpenAI's official Chat
        Completions API. The reasoning trace is preserved separately in
        `self.final_output["reasoning_summary"]` for the caller.
        """
        input_messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": fcd["id"],
                        "type": "function",
                        "function": {
                            "name": fcd["name"],
                            "arguments": fcd.get("arguments", "{}"),
                        },
                    }
                    for fcd in function_call_data_list
                ],
            }
        )

    def _append_tool_result(
        self,
        tool_call_id: str,
        content: str,
        input_messages: List[Dict[str, Any]],
    ) -> None:
        """Append one role:'tool' result message bound to its parent tool_call_id."""
        input_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
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
        reasoning_content = self._get_reasoning_content(message)
        content, inline_reasoning = self._split_inline_reasoning(content)
        if inline_reasoning:
            reasoning_content = (reasoning_content or "") + inline_reasoning
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
            input_messages = self.handle_function_calls(
                [
                    {
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
                input_messages,
            )
            input_messages = self._trim_messages_for_recursion(input_messages)
            self.ask_model(input_messages)
            return response.id

        if finish_reason == "length":
            self.final_output["truncated"] = True
        if finish_reason == "content_filter":
            self.final_output["filtered"] = True

        if not self._has_valid_content(content) and not reasoning_content:
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
        accumulated_partial_reasoning_parts = []
        received_any_content = False
        output_format = self.output_format_type
        index = 0
        reasoning_index = 0
        tool_call_count = 0
        finish_reason = None

        tool_call_accumulators: Dict[int, Dict[str, Any]] = {}

        if self.enable_timeline_log:
            stream_start_time = pendulum.now("UTC")

        first_delta_logged = False
        # Optional fallback for compatible servers that emit inline <think>
        # tags instead of the provider's dedicated reasoning_content field.
        # Off by default; enable via configuration.enable_think_tag_split.
        think_splitter = _ThinkTagSplitter() if self._enable_think_tag_split else None

        # WebSocket lifecycle state matching openai_agent_handler's pattern.
        content_message_started = False
        reasoning_active = False
        reasoning_no = 0

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

            # One-time diagnostic: log the keys present in the first delta so a
            # provider returning reasoning in a non-standard field is visible.
            if (
                not first_delta_logged
                and self.logger
                and self.logger.isEnabledFor(logging.INFO)
            ):
                first_delta_logged = True
                try:
                    delta_dump = delta.model_dump(exclude_none=True)
                    self.logger.info(
                        f"[STREAM_DELTA_KEYS] first delta keys={list(delta_dump.keys())} "
                        f"sample={_truncate(str(delta_dump))}"
                    )
                except Exception:
                    self.logger.info(
                        f"[STREAM_DELTA_KEYS] first delta attrs="
                        f"{[a for a in dir(delta) if not a.startswith('_')]}"
                    )

            if getattr(delta, "content", None):
                received_any_content = True
                # Optionally split any inline <think>...</think> blocks out
                # of content (opt-in via configuration.enable_think_tag_split).
                # Pass-through when disabled.
                if think_splitter is not None:
                    content_part, think_part = think_splitter.feed(delta.content)
                else:
                    content_part, think_part = delta.content, ""
                if think_part:
                    reasoning_active = True
                    print(think_part, end="", flush=True)
                    accumulated_reasoning_parts.append(think_part)
                    accumulated_partial_reasoning_parts.append(think_part)
                    reasoning_index, remaining_reasoning = self.process_text_content(
                        reasoning_index,
                        "".join(accumulated_partial_reasoning_parts),
                        "text",
                        suffix=f"rs#{reasoning_no}",
                    )
                    accumulated_partial_reasoning_parts = (
                        [remaining_reasoning] if remaining_reasoning else []
                    )
                if content_part:
                    # Transition from reasoning to content: flush leftover
                    # reasoning text and close the reasoning block before
                    # opening the content message.
                    if reasoning_active:
                        if accumulated_partial_reasoning_parts:
                            self.send_data_to_stream(
                                index=reasoning_index,
                                data_format="text",
                                chunk_delta="".join(
                                    accumulated_partial_reasoning_parts
                                ),
                                suffix=f"rs#{reasoning_no}",
                            )
                            accumulated_partial_reasoning_parts = []
                            reasoning_index += 1
                        reasoning_active = False
                        reasoning_no += 1

                    # Signal start of content message on the first content delta.
                    if not content_message_started:
                        content_message_started = True
                        if index == 0 and reasoning_index > 0:
                            index = reasoning_index + 1
                        self.send_data_to_stream(
                            index=index,
                            data_format=output_format,
                        )
                        index += 1

                    print(content_part, end="", flush=True)
                    accumulated_text_parts.append(content_part)

                    if output_format in ["json_object", "json_schema"]:
                        accumulated_partial_json_parts.append(content_part)
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
                        accumulated_partial_text_parts.append(content_part)
                        index, remaining_text = self.process_text_content(
                            index,
                            "".join(accumulated_partial_text_parts),
                            output_format,
                        )
                        accumulated_partial_text_parts = (
                            [remaining_text] if remaining_text else []
                        )

            # Providers use different field names for streamed reasoning chunks:
            #   SGLang / Qwen3 / OpenAI o-series / GLM via SGLang -> reasoning_content
            #   Anthropic Claude on some Bedrock paths             -> thinking
            #   Some misc proxies                                  -> reasoning
            reasoning_content = self._get_reasoning_content(delta)
            if reasoning_content:
                received_any_content = True
                reasoning_active = True
                print(reasoning_content, end="", flush=True)
                accumulated_reasoning_parts.append(reasoning_content)
                accumulated_partial_reasoning_parts.append(reasoning_content)
                reasoning_index, remaining_reasoning = self.process_text_content(
                    reasoning_index,
                    "".join(accumulated_partial_reasoning_parts),
                    "text",
                    suffix=f"rs#{reasoning_no}",
                )
                accumulated_partial_reasoning_parts = (
                    [remaining_reasoning] if remaining_reasoning else []
                )

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
                    # Close out any open reasoning / content message group on
                    # the wire before recursing into the tool-call round.
                    # Otherwise a model that emitted text + tool_calls in the
                    # same turn would leave the content message unterminated
                    # from the WebSocket consumer's perspective.
                    if reasoning_active and accumulated_partial_reasoning_parts:
                        self.send_data_to_stream(
                            index=reasoning_index,
                            data_format="text",
                            chunk_delta="".join(accumulated_partial_reasoning_parts),
                            suffix=f"rs#{reasoning_no}",
                        )
                        accumulated_partial_reasoning_parts = []
                        reasoning_index += 1
                        reasoning_active = False
                        reasoning_no += 1
                    if accumulated_partial_text_parts:
                        self.send_data_to_stream(
                            index=index,
                            data_format=output_format,
                            chunk_delta="".join(accumulated_partial_text_parts),
                        )
                        accumulated_partial_text_parts = []
                        index += 1
                    if content_message_started:
                        self.send_data_to_stream(
                            index=index,
                            data_format=output_format,
                            is_message_end=True,
                        )
                        content_message_started = False

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
                    input_messages = self.handle_function_calls(
                        tool_calls,
                        input_messages,
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

        # Drain anything still buffered in the think-tag splitter (e.g. a
        # final segment that arrived without ever crossing a tag boundary).
        # No-op when the splitter is disabled.
        if think_splitter is not None:
            tail_content, tail_think = think_splitter.flush()
        else:
            tail_content, tail_think = "", ""
        if tail_think:
            reasoning_active = True
            print(tail_think, end="", flush=True)
            accumulated_reasoning_parts.append(tail_think)
            accumulated_partial_reasoning_parts.append(tail_think)
            reasoning_index, remaining_reasoning = self.process_text_content(
                reasoning_index,
                "".join(accumulated_partial_reasoning_parts),
                "text",
                suffix=f"rs#{reasoning_no}",
            )
            accumulated_partial_reasoning_parts = (
                [remaining_reasoning] if remaining_reasoning else []
            )
        if tail_content:
            # Tail content forces the same reasoning-to-content transition we
            # do for in-flight deltas above.
            if reasoning_active:
                if accumulated_partial_reasoning_parts:
                    self.send_data_to_stream(
                        index=reasoning_index,
                        data_format="text",
                        chunk_delta="".join(accumulated_partial_reasoning_parts),
                        suffix=f"rs#{reasoning_no}",
                    )
                    accumulated_partial_reasoning_parts = []
                    reasoning_index += 1
                reasoning_active = False
                reasoning_no += 1
            if not content_message_started:
                content_message_started = True
                if index == 0 and reasoning_index > 0:
                    index = reasoning_index + 1
                self.send_data_to_stream(
                    index=index,
                    data_format=output_format,
                )
                index += 1
            print(tail_content, end="", flush=True)
            accumulated_text_parts.append(tail_content)
            if output_format not in ["json_object", "json_schema"]:
                accumulated_partial_text_parts.append(tail_content)

        # End-of-stream flush: leftover reasoning text, then leftover content
        # text, then the message-end signal. Mirrors the sibling's pattern of
        # `response.reasoning_summary_text.done` -> `response.output_text.done`
        # -> `response.content_part.done`.
        if reasoning_active and accumulated_partial_reasoning_parts:
            self.send_data_to_stream(
                index=reasoning_index,
                data_format="text",
                chunk_delta="".join(accumulated_partial_reasoning_parts),
                suffix=f"rs#{reasoning_no}",
            )
            accumulated_partial_reasoning_parts = []
            reasoning_index += 1
            reasoning_active = False
            reasoning_no += 1

        if accumulated_partial_text_parts:
            self.send_data_to_stream(
                index=index,
                data_format=output_format,
                chunk_delta="".join(accumulated_partial_text_parts),
            )
            accumulated_partial_text_parts = []
            index += 1

        if content_message_started:
            self.send_data_to_stream(
                index=index,
                data_format=output_format,
                is_message_end=True,
            )

        final_accumulated_text = "".join(accumulated_text_parts)

        if (
            self.enable_timeline_log
            and self.logger
            and self.logger.isEnabledFor(logging.INFO)
        ):
            stream_duration_ms = (
                pendulum.now("UTC") - stream_start_time
            ).total_seconds() * 1000
            elapsed = self._get_elapsed_time()
            self.logger.info(
                f"[TIMELINE] T+{elapsed:.2f}ms: Stream completed, run_id={run_id} "
                f"(took {stream_duration_ms:.2f}ms from stream start)"
            )

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
