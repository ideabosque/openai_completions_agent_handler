# OpenAI Completions Agent Handler Development Plan

## 1. Purpose

`OpenAICompletionsEventHandler` is a SilvaEngine event handler for OpenAI-compatible Chat Completions APIs. It intentionally calls `client.chat.completions.create()` rather than the newer Responses API so the same handler can run against OpenAI, SGLang, vLLM, LiteLLM, and other providers that expose the standard Chat Completions surface.

The v1 goal is a production-ready synchronous handler that can:

- Build valid Chat Completions `messages` payloads from SilvaEngine agent inputs.
- Inject the agent's `instructions` per request as a `system`, `developer`, or merged-into-first-user message.
- Support text input, caller-supplied Chat Completions content arrays, function tools, streaming, and non-streaming output.
- Continue conversations after tool calls using strict Chat Completions assistant/tool history, including the single-assistant batched shape for parallel tool calls.
- Pass provider-specific options through `extra_body`, with shorthand support for `enable_thinking` and `separate_reasoning`.
- Provide predictable resource cleanup through `close()` and context-manager usage.
- Emit structured `[MODEL_CALL]`, `[TOOL_CALL]`, and `[TOOL_RESULT]` log lines for production observability.
- Stay deterministic under local tests without OpenAI credentials, live model calls, or network access.

## 2. Current Project Assessment

The core handler is substantially implemented. A deterministic unit suite and two interactive smoke scripts exist locally under `openai_completions_agent_handler/tests`, but that directory is intentionally ignored by `.gitignore` for now and is not tracked by Git. Treat those tests as local validation assets, not release artifacts.

### Implemented

- Pooled `httpx.Client` with explicit timeouts and limits.
- OpenAI SDK client construction with `base_url`, `api_key`, and SDK-level `max_retries`.
- `close()` and context-manager (`__enter__` / `__exit__`) for resource cleanup.
- `enabled_tools` filtering at construction time.
- `max_completion_tokens` precedence over `max_tokens` with a warning.
- `_assemble_extra_body()` shorthand merge for `enable_thinking` and `separate_reasoning`; explicit `extra_body` values win.
- `_merge_instructions_into_first_user()` for `instructions_role="user"`, useful for Gemma-family chat templates that reject system messages.
- `invoke_model(**kwargs)` with inline payload assembly, instruction injection, `stream_options` handling, and `_omit_none()` cleanup.
- Non-streaming `handle_response()` for text, tool calls, `length`, `content_filter`, empty-response retries, and `reasoning_content`.
- Streaming `handle_stream()` with text accumulation, tool-call delta stitching by `index`, usage-only chunks, reasoning deltas, and finish-reason handling.
- Live stream rendering through `print(delta.content, end="", flush=True)` for local development.
- `handle_function_calls()` appends one assistant message containing all tool calls from a model turn, then appends one tool result per call.
- `_append_assistant_with_tool_calls()` and `_append_tool_result()` as focused history helpers.
- `[MODEL_CALL]`, `[TOOL_CALL]`, and `[TOOL_RESULT]` INFO logs with bounded argument/output previews.
- `ToolCallDepthExceeded` for bounded recursion.
- `_trim_messages_for_recursion()` with a leading-tool guard and explicit handling for `instructions_role="user"`.
- API-key redaction through `_redact_api_key()`.
- SDK exception propagation: `invoke_model()` and `ask_model()` re-raise `openai.APIError` subclasses and `ToolCallDepthExceeded` without wrapping.
- Minimal image content-array helpers: `_validate_image_url()` and `_build_image_message()`.

### Known Gaps Before v1

- **Tests are intentionally local-only for now.** `openai_completions_agent_handler/tests` exists on disk, but `git ls-files` does not include it and `.gitignore` intentionally ignores `tests/`. The README Quick Start should not imply that a fresh clone from tracked files alone can run `openai_completions_agent_handler.tests.test_deterministic` unless the tests are distributed through another documented path.
- **`_short_term_memory` writes assume `self.agent["tool_call_role"]`.** That key is read from the top-level agent dict, not `configuration`, and is not covered by `configuration_schema.json`.
- **Test coverage is partial.** The local deterministic suite now covers exception propagation, image helpers, `_assemble_extra_body` deep merge, `_merge_instructions_into_first_user` shapes, `handle_function_calls()` parallel batching, and `[MODEL_CALL]` / `[TOOL_CALL]` / `[TOOL_RESULT]` log emission shape. Remaining gaps: split streaming tool-call chunks across more than two delta arrivals, final usage-only stream chunk with empty `choices`, and end-to-end recursion-depth lifecycle that walks through more than one tool call before hitting the cap.
- **Interactive smoke tests rely on env-driven config.** `.env.example` documents the local-provider defaults, but Gemma's required `instructions_role="user"` is easy to miss.
- **Dependency install path is fragile.** `pyproject.toml` depends on `AI-Agent-Handler` and `SilvaEngine-Utility`, while the README Quick Start mentions `AI-Agent-Handler` and `ai_agent_core_engine`. The docs should identify the complete dependency source of truth.
- **Packaging includes broad package discovery.** `tool.setuptools.packages.find.include = ["*"]` can accidentally include generated or unrelated package directories.

## 3. v1 Scope

### In Scope

- Synchronous `OpenAICompletionsEventHandler`.
- Chat Completions request construction and invocation.
- Streaming and non-streaming response handling, including live stdout chunk rendering for local development.
- Function tool calls, including batched assistant `tool_calls` for parallel tool calls, and recursive continuation after tool results.
- Three instruction-injection modes: leading `system`, leading `developer`, and merged-into-first-user.
- Caller-supplied multimodal Chat Completions content arrays, plus `_build_image_message()` as a small convenience helper for `http`, `https`, and `data` image URLs.
- Provider extension passthrough through `extra_body`, plus shorthand keys for common SGLang/Qwen thinking settings.
- Explicit configuration behavior documented by `configuration_schema.json`.
- Local deterministic tests using mocked clients and mocked stream chunks.
- README examples that match the actual constructor, configuration shape, and message formats.
- Structured observability logging at INFO level.

### Out of Scope

These are Responses API or platform-specific capabilities and remain in the sibling Responses-based handler:

- Native `web_search`, `code_interpreter`, MCP-as-native-tools, and MCP approval flows.
- Responses API item formats such as `function_call_output`.
- Container file citations or `get_output_file()` backed by OpenAI container endpoints.
- Server-side conversation state.
- Async `AsyncOpenAI` support or async streaming.

### Not Included in v1

- File-driven image ingestion: `_process_input_files`, `_process_user_file_ids`, an `input_files` parameter on `ask_model`, file upload helpers, and automatic local-file-to-data-URL conversion.
- Audio input/output.
- Inline PDF or arbitrary file content parts.
- `n > 1` candidate sampling.
- `logprobs` and `top_logprobs` capture.
- Prompt caching cost reporting.
- Shared handler instances across concurrent request trees.

## 4. API Contract

The handler must remain aligned with Chat Completions semantics:

| Area | Required Behavior |
|---|---|
| SDK call | `client.chat.completions.create()` |
| Request input | `messages`, not `input` |
| Multimodal input | Caller-supplied content arrays, for example text parts plus `image_url` parts |
| Assistant tool call | One assistant message with `content: None` and `tool_calls` containing all tool calls from the model turn |
| Tool result | One `{"role": "tool", "tool_call_id": "...", "content": "..."}` message per tool call |
| Streaming text | Read `choices[0].delta.content` |
| Streaming tools | Stitch `choices[0].delta.tool_calls` by `index` |
| Usage in streams | Handle usage-only chunks where `choices == []` |
| Conversation state | Caller-managed message history only |

Payload hygiene is part of the contract:

- `_omit_none()` strips `None` values before the SDK call.
- `stream_options` is injected only when `stream=True`.
- `tools`, `tool_choice`, and `parallel_tool_calls` are removed when the filtered tool list is empty.
- Image helper URLs are limited to `http`, `https`, and `data` schemes; no local file paths are read by the handler.
- Provider-specific parameters belong in `extra_body`.
- Tool-call continuation must preserve the assistant tool-call message and all matching tool result messages as a complete group.

## 5. Configuration Design

`configuration_schema.json` is the source of truth for supported configuration fields.

Required fields:

- `model`
- `openai_api_key`

Important optional fields:

- `base_url`
- `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`
- `max_tokens`, `max_completion_tokens`
- `tools`, `tool_choice`, `parallel_tool_calls`
- `response_format`
- `stop`, `seed`, `logit_bias`
- `instructions_role`: `"system"` (default), `"developer"`, or `"user"`
- `enabled_tools`
- `extra_body`
- `enable_thinking`, `separate_reasoning`
- `request_timeout_seconds`, `connect_timeout_seconds`
- `max_retries`
- `pool_max_connections`, `pool_max_keepalive_connections`
- `max_tool_call_depth`

Configuration rules:

- If both `max_tokens` and `max_completion_tokens` are configured, keep `max_completion_tokens`, remove `max_tokens`, and log a warning.
- `instructions_role="user"` prepends instructions to the first user message at payload-build time instead of storing instructions in history.
- Provider-specific settings must live under `extra_body`. The shorthand keys `enable_thinking` and `separate_reasoning` populate `extra_body`, and explicit `extra_body` values win.
- `openai_api_key` is never logged.
- Tool arguments and full message bodies must not be logged at INFO level by default.
- `stream_options` is not user-facing configuration.

## 6. Provider Compatibility Notes

### OpenAI

- Use the default SDK `base_url` by omitting `base_url`.
- Prefer `max_completion_tokens` for newer models.
- Use `instructions_role="system"` unless the target model requires `developer`.
- Responses-only features are intentionally unsupported in this handler.

### vLLM

- Launch: `vllm serve <model> [--api-key token-abc123] [--chat-template ./template.jinja]`
- Base URL: `http://localhost:8000/v1`
- Authentication is optional for local servers.
- Provider-specific parameters should pass through `extra_body`.
- Models without a tokenizer chat template need `--chat-template`.
- Image input: supports the standard `image_url` content-part format used by `ask_model(input_images=...)`, but documented as **one image per message** on stable releases. Multi-image support is version-dependent; confirm against your vLLM build before sending more than one URL per turn.

References: [vLLM OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/), [vLLM Multimodal Inputs](https://docs.vllm.ai/en/stable/features/multimodal_inputs/)

### SGLang

- Launch example: `python3 -m sglang.launch_server --model-path Qwen/Qwen3-4B --tool-call-parser qwen --reasoning-parser qwen3`
- Base URL: `http://127.0.0.1:30000/v1`
- Use `openai_api_key="EMPTY"` for local servers that skip auth.
- For reasoning support, set `enable_thinking=true` and `separate_reasoning=true`, or set `extra_body.chat_template_kwargs` directly.
- Function calling requires a matching `--tool-call-parser`; otherwise tool calls may arrive as raw text.
- Gemma-family models should use `instructions_role="user"` and should be treated as chat-only through SGLang unless tool support is explicitly confirmed for that stack.
- Image input: supports the standard `image_url` content-part format, including multiple image parts per message in recent versions. Requires launching with a vision-capable model (`Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen/Qwen3-VL-*`, etc.); chat-only models will reject image content. Older builds had Qwen2.5-VL-72B parsing bugs (resolved upstream).

References: [SGLang OpenAI API](https://docs.sglang.io/docs/basic_usage/openai_api), [SGLang Tool and Function Calling](https://docs.sglang.ai/advanced_features/function_calling.html), [SGLang OpenAI Vision API](https://docs.sglang.io/basic_usage/openai_api_vision.html)

### LiteLLM

- Launch: `litellm --config <config.yaml>`
- Base URL: `http://localhost:4000`
- The handler treats LiteLLM as a standard OpenAI-compatible endpoint; provider-specific options pass through `extra_body`.

## 7. v1 Roadmap

Done items are listed in section 2. The roadmap below contains open work.

### Phase 1: Local Test Coverage and Documentation

Goal: keep the local deterministic suite current while documenting that it is not part of tracked release contents yet.

Tasks:

- Keep `openai_completions_agent_handler/tests` ignored for now.
- Update the README Quick Start to make the local-test prerequisite explicit, or move the test command into a maintainer-only validation section.
- Add streaming tests for tool-call deltas split across three or more chunks (single delta covered, multi-delta gap remains).
- Add a test for the final usage-only stream chunk where `choices == []`.
- Add an end-to-end recursion-depth lifecycle test that walks through one or more real tool-call rounds before hitting the cap.
- Add a test that wrapped (non-API) exception messages do not leak the API key.

Acceptance gate:

- Maintainer-local checkout with the ignored test bundle can run `python -m unittest openai_completions_agent_handler.tests.test_deterministic -v` without API credentials or network access.
- README wording clearly distinguishes install instructions for package users from maintainer-local validation commands.

### Phase 2: Exception Propagation

Completed. `invoke_model()` and `ask_model()` now re-raise `openai.APIError` subclasses and `ToolCallDepthExceeded` without conversion. Unknown exceptions are still wrapped with redacted messages.

### Phase 3: Packaging and Install Path

Goal: a clean checkout can be installed and tested without insider knowledge.

Tasks:

- Document where `AI-Agent-Handler`, `SilvaEngine-Utility`, and any runtime coupling to `ai_agent_core_engine` come from.
- Update the README Quick Start so it matches `pyproject.toml` and the actual smoke-test dependencies.
- Narrow `tool.setuptools.packages.find.include` so only `openai_completions_agent_handler*` is packaged.
- Keep `.gitignore` hiding `tests/`, `.env`, caches, build output, and egg metadata.
- Consider adding a `[project.optional-dependencies]` `dev` group for deterministic and smoke-test dependencies.

Acceptance gate:

- A fresh clone, following only README instructions, can install the package and run the deterministic tests.

### Phase 4: Documentation Drift Cleanup

Goal: align README, development plan, schema, and code on the same source of truth.

Tasks:

- Verify every documented config field appears in `configuration_schema.json` and vice versa.
- Document the top-level `tool_call_role` agent key, or remove the direct dependency on it.
- Add an "Observability" README section for `[MODEL_CALL]`, `[TOOL_CALL]`, and `[TOOL_RESULT]`.
- Add a "Streaming" README section explaining stdout chunk rendering and any production streaming path.
- Refresh the production checklist for `extra_body` shorthand, parallel tool calls, and handler cleanup.

Acceptance gate:

- A new contributor can read the README and schema and accurately predict runtime behavior for every documented field.

### Phase 5: Release Polish

Goal: tag a v1 release.

Tasks:

- Bump version in `pyproject.toml`.
- Confirm all Phase 1-4 acceptance gates pass.
- Run the manual smoke matrix.
- Write release notes covering parallel tool-call handling, Gemma support, SGLang shorthand, structured logging, exception propagation, and image helper scope.

## 8. Test Strategy

### 8.1 Deterministic Suite

This is the maintainer-local release gate while `tests/` remains ignored. It must not require network access, API keys, real model calls, or interactive scripts.

Required coverage:

- Constructor and OpenAI client setup.
- `httpx.Client` timeout and pool configuration.
- `close()` and context-manager cleanup.
- Payload construction and `_omit_none()` stripping.
- `instructions_role` behavior for `system`, `developer`, and `user`.
- `_assemble_extra_body` shorthand assembly.
- `max_completion_tokens` precedence over `max_tokens`.
- `enabled_tools` filtering.
- `extra_body` and `response_format` passthrough.
- Non-streaming response handling for `stop`, `tool_calls`, `length`, `content_filter`, and unknown finish reasons.
- Streaming response handling for text accumulation, tool-call stitching, reasoning deltas, and usage-only chunks.
- `handle_function_calls()` single and parallel tool-call shapes.
- Recursion depth through `ToolCallDepthExceeded`.
- Message trimming integrity, including the orphan-tool guard.
- Empty-response retry caps.
- Secret redaction in errors and logs.
- SDK exception propagation from both `invoke_model()` and `ask_model()`.
- Image helper URL validation and content-array shape.
- Structured log emission shape.

### 8.2 Interactive Smoke Scripts

The local working directory currently contains:

- [test_chatbot.py](openai_completions_agent_handler/tests/test_chatbot.py): interactive REPL for streaming and non-streaming chat.
- [test_tool_call.py](openai_completions_agent_handler/tests/test_tool_call.py): single-turn tool-call exercise with a local Python function dispatcher.

Note: these scripts are present locally today and intentionally remain untracked while `tests/` is ignored.

### 8.3 Manual Smoke Matrix

Run only after the deterministic suite passes:

- OpenAI text completion.
- OpenAI function call.
- OpenAI streaming function call.
- SGLang + Qwen3 through `base_url` with `--tool-call-parser qwen --reasoning-parser qwen3`.
- SGLang + Gemma through `base_url` with `instructions_role=user` for chat-only validation.
- vLLM through `base_url`.
- LiteLLM proxy through `base_url`.
- Parallel tool calls: verify the on-wire payload has one assistant message with multiple `tool_calls`.

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| README references ignored local tests | Fresh clones cannot run documented validation | Mark test commands as maintainer-local or document the separate test-bundle source |
| Streaming tool deltas stitched incorrectly | Broken function names or malformed JSON | Chunk-level stream tests by `index` |
| Usage-only stream chunks dereferenced as choices | Crash at stream end | Explicit `choices == []` tests |
| Empty model output retries forever | Hang or runaway spend | Hard retry cap |
| `_ask_model_depth` leaks after exceptions | Later calls fail unexpectedly | `finally` decrement and depth tests |
| API key appears in logs or exceptions | Secret disclosure | Redaction helper and log tests |
| SDK exceptions wrapped generically | Callers cannot handle retryable failures | Preserve original exception types |
| Parallel tool calls emitted as separate assistants | Stricter providers reject continuation | Batched assistant message tests |
| Gemma rejects system role | Provider 400 from chat template | `instructions_role="user"` tests and docs |
| SGLang parser not configured | Tool calls arrive as raw text | Document required `--tool-call-parser` |
| Provider parameter support drifts | Provider-specific failures | Omit unset values and isolate `extra_body` |
| README examples drift from code | User integration failures | Validate examples during release |
| Transitive dependencies unavailable | Install fails in clean environments | Document local/Git/PyPI dependency source |
| Broad package discovery includes unwanted files | Dirty release artifacts | Narrow setuptools discovery |

## 10. Release Checklist

Before tagging v1:

- README distinguishes package-user setup from maintainer-local validation with the ignored test bundle.
- Deterministic tests pass in the maintainer-local checkout.
- SDK exception types propagate unchanged from both `invoke_model()` and `ask_model()`.
- The package installs from a clean checkout using the documented dependency flow.
- No `__pycache__`, test cache, egg metadata, or build artifacts are included in release artifacts.
- `configuration_schema.json` matches implemented behavior.
- README examples match actual constructor arguments and message shapes.
- Manual smoke matrix passes for OpenAI, SGLang+Qwen3, and one of vLLM/LiteLLM.
- Both streaming and non-streaming tool-call paths are exercised end to end.
- Parallel tool-call payloads are inspected and confirm one batched assistant message.
- Error logs are inspected for API-key redaction.

## 11. Recommended Next Work Items

In suggested execution order:

1. Update README Quick Start so ignored local tests are presented as maintainer-local validation, not a fresh-clone guarantee.
2. Fill the remaining test gaps: multi-chunk streaming tool-call stitching, final usage-only stream chunk, end-to-end recursion-depth lifecycle, and API-key redaction on wrapped exceptions.
3. Update README dependency instructions so they match `pyproject.toml` and the actual local-provider runtime dependencies.
4. Narrow setuptools package discovery.
5. Decide whether `tool_call_role` belongs in a top-level agent schema/docs section or should be made optional in code.
6. Run the manual smoke matrix against OpenAI and at least one local provider.
7. Tag v1.
