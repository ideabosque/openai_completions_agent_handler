# OpenAI Completions Agent Handler - Development Plan

## 1. Purpose

`OpenAICompletionsEventHandler` is a SilvaEngine event handler for OpenAI-compatible Chat Completions APIs. It uses `client.chat.completions.create()` instead of the newer Responses API so the same handler can run against OpenAI, SGLang, vLLM, LiteLLM, and other servers that expose the standard Chat Completions surface.

The project goal is a production-ready synchronous handler that can:

- Build valid Chat Completions `messages` payloads from SilvaEngine agent inputs.
- Support text, image URL/data URL inputs, function tools, streaming, and non-streaming output.
- Continue conversations after tool calls using the Chat Completions assistant/tool message format.
- Pass provider-specific options through `extra_body` without polluting core control flow.
- Remain deterministic under tests without requiring real OpenAI credentials or network access.

## 2. Current Project Assessment

Current implementation status, based on the repository state:

- Package metadata, schema, README, implementation module, and deterministic tests exist.
- The handler already initializes an `httpx.Client`, creates an OpenAI SDK client, filters enabled tools, applies `max_completion_tokens` precedence, and exposes `close()` plus context manager support.
- Non-streaming response handling, streaming response handling, tool-call continuation, message trimming, image content-array conversion, file helpers, and basic logging are implemented.
- The test suite covers many expected v1 behaviors, but it cannot currently run in a clean environment unless dependencies such as `openai` are installed.
- The README has mojibake/encoding artifacts in headings and icons. That should be cleaned before release because it lowers package quality even if the code works.

The previous plan was technically useful but read like a first-build implementation spec. This revised plan treats the project as partially implemented and focuses on getting it to a reliable v1 release.

## 3. Scope

### In Scope for v1

- Synchronous `OpenAICompletionsEventHandler` only.
- Chat Completions request construction and invocation.
- Streaming and non-streaming response handling.
- Function tool calls and recursive model continuation after tool results.
- Image input via Chat Completions content arrays using `image_url` parts.
- Provider extension passthrough through `extra_body`.
- File helper methods where compatible with the OpenAI SDK file API.
- Deterministic unit tests using mocked clients and mocked stream chunks.
- README, schema, and examples aligned with the actual implementation.

### Out of Scope for v1

These are Responses API or platform-specific capabilities and should remain in the sibling Responses-based handler:

- Native `web_search`, `code_interpreter`, MCP tools, and MCP approval flows.
- Responses API item formats such as `function_call_output`.
- Container file citations or `get_output_file()` backed by container endpoints.
- Server-side conversation state.
- Async `AsyncOpenAI` support or async streaming.

### Deferred Enhancements

- Audio input/output.
- Inline PDF or arbitrary file content parts.
- `n > 1` candidate sampling.
- `logprobs` and `top_logprobs` capture.
- Prompt caching cost reporting.
- Shared handler instances across concurrent request trees.

## 4. API Contract

The handler must follow Chat Completions semantics exactly:

| Area | Required Chat Completions Behavior |
|---|---|
| SDK call | `client.chat.completions.create()` |
| Request input | `messages`, not `input` |
| Assistant tool call | Assistant message with `tool_calls` |
| Tool result | `{"role": "tool", "tool_call_id": "...", "content": "..."}` |
| Tool-call assistant content | `content: None`, not an empty string |
| Streaming text | `choices[0].delta.content` |
| Streaming tools | Stitch `choices[0].delta.tool_calls` by `index` |
| Usage in streams | Handle final usage-only chunks where `choices == []` |
| Conversation state | Caller-managed message history only |

Provider compatibility depends on clean payloads. Optional values must be omitted when unset, and tool-related options must only be sent when tools are present.

## 5. Configuration Design

`configuration_schema.json` should remain the source of truth for supported configuration fields.

Required core fields:

- `model`
- `openai_api_key`

Important optional fields:

- `base_url`
- `temperature`
- `top_p`
- `max_tokens`
- `max_completion_tokens`
- `presence_penalty`
- `frequency_penalty`
- `tools`
- `tool_choice`
- `parallel_tool_calls`
- `response_format`
- `stop`
- `seed`
- `logit_bias`
- `stream_options`
- `instructions_role`
- `enabled_tools`
- `extra_body`
- `request_timeout_seconds`
- `connect_timeout_seconds`
- `max_retries`
- `pool_max_connections`
- `pool_max_keepalive_connections`
- `max_tool_call_depth`

Configuration rules:

- If both `max_tokens` and `max_completion_tokens` are configured, prefer `max_completion_tokens` and log a warning.
- `instructions_role` must be explicit configuration, defaulting to `system`. Do not infer `developer` from model names.
- Provider-specific settings must live under `extra_body`.
- Never log `openai_api_key`.
- Avoid logging full message bodies or full tool arguments at INFO level.

## 6. v1 Completion Roadmap

### Phase 1 - Dependency and Test Harness Cleanup

Goal: make local validation predictable.

Tasks:

- Add a documented install/test command to the README.
- Ensure test dependencies are declared through packaging metadata or a clearly documented dev install path.
- Decide whether tests should run with `python -m unittest ...` or a pytest entrypoint, then document one official path.
- Remove committed `__pycache__` artifacts from the package tree and keep them ignored.

Acceptance gate:

- A fresh checkout can install dependencies and run deterministic tests without real API credentials.

### Phase 2 - Payload and Invocation Hardening

Goal: make requests portable across OpenAI-compatible providers.

Tasks:

- Confirm `_build_request_payload()` omits unset optional values.
- Only include `stream_options` when streaming.
- Only include `tools`, `tool_choice`, and `parallel_tool_calls` when a non-empty tool list is present.
- Pass `extra_body` through unchanged.
- Re-raise SDK connection, timeout, and rate-limit exceptions without wrapping them in a generic exception that hides their type.
- Log BadRequest context using safe metadata only: model, stream flag, has-tools, has-image, and configured response format.

Acceptance gate:

- Exact-payload tests cover streaming, non-streaming, no-tools, tools, `extra_body`, and `response_format`.

### Phase 3 - Message Construction and Trimming

Goal: guarantee that generated message history is valid Chat Completions input.

Tasks:

- Ensure agent instructions are inserted as the first `system` or `developer` message when applicable.
- Preserve existing assistant `tool_calls` and `role: "tool"` messages without rewriting IDs or order.
- Build image messages as content arrays with text and `image_url` parts.
- Reject unsupported image URL schemes at the boundary.
- Strengthen trimming so it never leaves an orphan `tool` message or partial tool-call group.

Acceptance gate:

- Tests cover text-only messages, text-plus-image messages, preserved prior tool history, and trimming through multi-tool-call chains.

### Phase 4 - Tool Call Lifecycle

Goal: make function execution reliable and recoverable.

Tasks:

- Validate every tool call has an `id`, a function name, and parseable arguments before execution.
- Parse JSON arguments exactly once.
- Persist tool-call status as `in_progress`, then `completed` or `failed`.
- Serialize tool outputs into string `content` before appending tool result messages.
- Continue the model with assistant-with-`tool_calls` followed by matching `tool` messages.
- Enforce `max_tool_call_depth` with `ToolCallDepthExceeded`.

Acceptance gate:

- Tests prove successful tool calls, failed tool calls, malformed arguments, multi-tool ordering, and recursion depth limits.

### Phase 5 - Response Handling

Goal: return clear final output in all supported finish states.

Tasks:

- Handle finish reasons: `stop`, `tool_calls`, `length`, `content_filter`, deprecated `function_call`, and unknown values.
- Capture `reasoning_content` defensively with `getattr`.
- Mark truncated and filtered outputs on `final_output`.
- Bound empty-response retries with a configurable or documented cap.
- Preserve usage and provider request ID for observability.

Acceptance gate:

- Tests cover text-only, tool-only, text-plus-tool, empty retry cap, reasoning content, truncation, filtering, and unknown finish reasons.

### Phase 6 - Streaming Reliability

Goal: make streaming behavior equivalent to non-streaming behavior.

Tasks:

- Accumulate text in lists and join once.
- Send stream deltas to the queue in order.
- Stitch streaming tool calls by `index`, including partial `id`, `function.name`, and `function.arguments`.
- Handle usage-only chunks where `choices` is empty.
- Finalize `stream_event` on all successful terminal paths.
- For streaming JSON, emit raw deltas and parse the final accumulated text once.

Acceptance gate:

- Tests cover text streaming, JSON streaming, usage-only chunks, parallel tool-call stitching, partial JSON arguments, `length`, and `content_filter`.

### Phase 7 - Documentation and Release Polish

Goal: make the project understandable and publishable.

Tasks:

- Fix README encoding artifacts.
- Explain why the handler targets Chat Completions despite OpenAI's newer Responses API.
- Provide minimal OpenAI, SGLang, vLLM/LiteLLM, function-tool, streaming, and image examples.
- Document unsupported Responses-only features and point users to the sibling handler.
- Add a production checklist: pin SDK versions, configure timeouts, set tool depth cap, redact logs, and close the handler.
- Add troubleshooting notes for invalid tool-call history and provider-specific unsupported parameters.

Acceptance gate:

- README examples match the actual constructor and method signatures.
- A user can understand when to choose this handler versus the Responses-based sibling.

## 7. Test Strategy

The deterministic test suite should remain the primary release gate. Tests should not require network access, API keys, real model calls, or interactive scripts.

Required coverage:

- Constructor and OpenAI client setup.
- `httpx.Client` timeout and pool configuration.
- `close()` and context manager cleanup.
- Payload construction and omission of unset values.
- `max_completion_tokens` precedence over `max_tokens`.
- Tool filtering with `enabled_tools`.
- Non-streaming response handling.
- Streaming response handling.
- Tool-call execution and continuation.
- Recursion depth cap.
- Message trimming integrity.
- Image content-array construction.
- Rejection of unsupported image schemes.
- Structured output passthrough.
- Secret redaction in errors and logs.

Manual smoke tests are secondary and should be used only after deterministic tests pass:

- OpenAI text completion.
- OpenAI function call.
- OpenAI streaming function call.
- SGLang through `base_url`.
- vLLM through `base_url`.
- LiteLLM proxy through `base_url`.
- Image-capable model with `data:` and `https:` image URLs.

## 8. Known Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Tool-call message history becomes invalid | Provider rejects continuation | Exact message-shape tests |
| Streaming tool deltas are stitched incorrectly | Broken function names or malformed JSON | Chunk-level stream tests by index |
| Usage-only stream chunk is dereferenced as a choice | Crash at end of stream | Explicit `choices == []` branch |
| Empty model output retries forever | Hang or runaway spend | Hard retry cap |
| `_ask_model_depth` leaks after exception | Broken recursion behavior | `finally` decrement and tests |
| API key appears in logs or exceptions | Secret disclosure | Redaction helper and log tests |
| Provider parameter support drifts | Provider-specific failures | Omit unset values and isolate `extra_body` |
| README examples drift from code | User integration failures | Example validation during release |
| Dependencies are missing in clean environments | Tests fail before exercising code | Documented install/test workflow |

## 9. Release Checklist

Before tagging v1:

- All deterministic tests pass locally.
- README encoding artifacts are fixed.
- `docs/development-plan.md` reflects actual project status.
- No `__pycache__` files are included in release artifacts.
- Configuration schema matches implemented configuration behavior.
- README examples use valid constructor arguments and message shapes.
- Manual smoke tests pass for at least one OpenAI model and one OpenAI-compatible local/proxy provider.

## 10. Suggested Next Work Items

Recommended order from the current repository state:

1. Fix packaging/dev dependency setup so tests run cleanly in a fresh environment.
2. Clean README encoding artifacts and align examples with the actual package.
3. Expand exact-payload tests for `invoke_model()` and `_build_request_payload()`.
4. Harden exception propagation so SDK exception types are not unnecessarily hidden.
5. Strengthen message trimming tests for multi-tool-call groups.
6. Add streaming JSON final-parse behavior or document that the current queue helpers own partial JSON processing.
7. Run manual smoke tests against OpenAI and at least one `base_url` provider.
