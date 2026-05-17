# OpenAICompletionsEventHandler

A concrete implementation of [`AIAgentEventHandler`](https://github.com/ideabosque/ai_agent_handler) that drives OpenAI's **Chat Completions API** (`client.chat.completions.create()`). It handles message construction, model invocation, function tools, streaming, and conversation continuation for the SilvaEngine AI agent pipeline.

Compatible with **OpenAI**, **SGLang**, **vLLM**, **LiteLLM**, and any other server that implements the standard Chat Completions API.

For the chosen scope, design rationale, and roadmap, see [docs/development-plan.md](docs/development-plan.md).

---

## Why Chat Completions instead of the Responses API?

OpenAI currently recommends the Responses API for new OpenAI-native agent applications. This handler intentionally targets Chat Completions because it is the common surface exposed by SGLang, vLLM, LiteLLM, and most OpenAI-compatible inference servers. If you need Responses-only features (`code_interpreter`, `web_search`, native MCP tool handling, container files), use the sibling [`openai_agent_handler`](https://github.com/ideabosque/openai_agent_handler) project instead.

---

## Inheritance

```
AIAgentEventHandler
     |
     +-- OpenAICompletionsEventHandler
```

---

## Public Surface

### Key attributes

| Attribute | Description |
|---|---|
| `client` | OpenAI SDK client; honors `base_url` for custom servers. |
| `model_setting` | Dict built from `agent["configuration"]`; passed to `chat.completions.create()` after `omit_none()` filtering. |
| `_http_client` | Pooled `httpx.Client` with configured timeouts and connection limits. |
| `instructions_role` | `"system"` (default) or `"developer"`. Configured explicitly; never inferred from model name. |
| `final_output` | Populated with `message_id`, `role`, `content`, and optional `reasoning_summary`, `truncated`, `filtered` flags. |

### Core methods

```python
def __init__(
    self,
    logger: logging.Logger,
    agent: Dict[str, Any],
    **setting: Dict[str, Any],
) -> None: ...

def invoke_model(
    self,
    messages: List[Dict[str, Any]],
    stream: bool = False,
    **kwargs,
) -> Any: ...

def ask_model(
    self,
    input_messages: List[Dict[str, Any]],
    queue: Queue = None,
    stream_event: threading.Event = None,
    input_files: Optional[List[Dict[str, Any]]] = None,
    model_setting: Dict[str, Any] = None,
) -> Optional[str]: ...

def close(self) -> None: ...
```

`invoke_model` reads all OpenAI parameters from `self.model_setting`; pass overrides through `**kwargs` only when you need a one-off change. The handler also supports the context-manager protocol so the pooled HTTP client is released cleanly.

---

## Sample Configuration

```json
{
  "endpoint_id": "openai-completions",
  "agent_name": "Weather Assistant",
  "instructions": "You are an AI Assistant. Use `get_weather_forecast` to answer weather queries. Always clarify ambiguous input.",
  "num_of_messages": 30,
  "tool_call_role": "developer",
  "configuration": {
    "model": "gpt-4o",
    "openai_api_key": "${OPENAI_API_KEY}",
    "temperature": 0,
    "max_completion_tokens": 5000,
    "instructions_role": "system",
    "max_tool_call_depth": 8,
    "request_timeout_seconds": 120.0,
    "connect_timeout_seconds": 10.0,
    "max_retries": 2,
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather_forecast",
          "description": "Get the weather forecast for a given city and date",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string", "description": "City name"},
              "date": {"type": "string", "description": "Forecast date (YYYY-MM-DD)"}
            },
            "additionalProperties": false,
            "required": ["city", "date"]
          }
        }
      }
    ]
  },
  "functions": {
    "get_weather_forecast": {
      "class_name": "WeatherForecastFunction",
      "module_name": "weather_funct",
      "configuration": {}
    }
  },
  "function_configuration": {
    "weather_provider": "open-meteo"
  }
}
```

Notes on the configuration:

- `instructions_role` is in `configuration` and controls the role of the leading instruction message. Default is `system`; set `developer` only when the target OpenAI model expects it.
- `tool_call_role` lives at the top level of the agent dict (not under `configuration`). It is the role used when the handler appends tool-call records to `_short_term_memory` for clients that read it back.
- If both `max_tokens` and `max_completion_tokens` are set, the handler keeps `max_completion_tokens` and warns once.
- `enabled_tools` (array of names) filters the `tools` list at construction time. Tools not in the set are dropped.
- `extra_body` is forwarded verbatim and is the only supported channel for provider-specific extensions.

### SGLang Configuration

```json
{
  "endpoint_id": "sglang-local",
  "agent_name": "Qwen Assistant",
  "configuration": {
    "model": "Qwen/Qwen3-4B",
    "base_url": "http://127.0.0.1:30000/v1",
    "openai_api_key": "EMPTY",
    "temperature": 0.7,
    "max_tokens": 1024,
    "extra_body": {
      "chat_template_kwargs": {"enable_thinking": true},
      "separate_reasoning": true
    }
  }
}
```

Use `"openai_api_key": "EMPTY"` for SGLang/vLLM servers that do not require authentication. The full list of supported configuration fields is in [`configuration_schema.json`](openai_completions_agent_handler/configuration_schema.json).

---

## Usage

### Non-Streaming Chatbot

```python
import pendulum
from openai_completions_agent_handler import OpenAICompletionsEventHandler

weather_agent = {...}  # see Sample Configuration

def latest_messages(memory, num):
    return [m["message"] for m in sorted(memory, key=lambda x: x["created_at"], reverse=True)][:num][::-1]

with OpenAICompletionsEventHandler(logger=None, agent=weather_agent) as handler:
    handler.short_term_memory = []
    while True:
        user_input = input("User: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        handler.short_term_memory.append({
            "message": {"role": "user", "content": user_input},
            "created_at": pendulum.now("UTC"),
        })
        messages = latest_messages(handler.short_term_memory, weather_agent["num_of_messages"])
        handler.ask_model(messages)
        print("Chatbot:", handler.final_output["content"])
        handler.short_term_memory.append({
            "message": handler.final_output,
            "created_at": pendulum.now("UTC"),
        })
```

The `**setting` kwargs accepted by the constructor are used by the base `AIAgentEventHandler` for SilvaEngine deployment concerns (AWS credentials for Lambda/S3 calls, S3-based function loading, message-invoker bootstrap). They are optional for a local quickstart: omit them and the base class falls back to default boto3 credentials and skips S3 function loading. Add them only when your deployment actually needs them — for example, `region_name` / `aws_access_key_id` / `aws_secret_access_key` for non-default AWS credentials, and `funct_bucket_name` / `funct_zip_path` / `funct_extract_path` when tool functions are packaged in S3.

### Streaming Chatbot

```python
import threading
import pendulum
from queue import Queue
from openai_completions_agent_handler import OpenAICompletionsEventHandler

weather_agent = {...}  # see Sample Configuration

def latest_messages(memory, num):
    return [m["message"] for m in sorted(memory, key=lambda x: x["created_at"], reverse=True)][:num][::-1]

with OpenAICompletionsEventHandler(logger=None, agent=weather_agent) as handler:
    handler.short_term_memory = []
    while True:
        user_input = input("User: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        handler.short_term_memory.append({
            "message": {"role": "user", "content": user_input},
            "created_at": pendulum.now("UTC"),
        })
        messages = latest_messages(handler.short_term_memory, weather_agent["num_of_messages"])

        stream_queue = Queue()
        stream_event = threading.Event()
        threading.Thread(
            target=handler.ask_model,
            args=(messages, stream_queue, stream_event),
            daemon=True,
        ).start()

        first = stream_queue.get()
        if first["name"] == "run_id":
            print("Run ID:", first["value"])

        stream_event.wait()
        print("Chatbot:", handler.final_output["content"])
        handler.short_term_memory.append({
            "message": handler.final_output,
            "created_at": pendulum.now("UTC"),
        })
```

### Image Input

Pass image inputs through `ask_model(input_files=...)`. Accepted shapes:

```python
handler.ask_model(
    input_messages=[{"role": "user", "content": "Describe this picture."}],
    input_files=[
        {"encoded_image": "<base64 of jpeg/png>"},   # converted to data: URL
        {"image_url": "https://example.com/cat.png"} # passed through as-is
    ],
)
```

Schemes other than `http://`, `https://`, and `data:` are rejected at the boundary. `https:` URLs are fetched by the model provider, not by this handler — see the security note below.

---

## Key Differences from `openai_agent_handler`

| Feature | `openai_agent_handler` (Responses API) | This handler (Chat Completions API) |
|---|---|---|
| API endpoint | `client.responses.create()` | `client.chat.completions.create()` |
| Input format | `input` list of items | `messages` list of role/content dicts |
| Tool result format | `function_call_output` items | `{"role": "tool", "tool_call_id": ..., "content": ...}` |
| Assistant tool call | `type: "function_call"` items | Assistant message with `tool_calls` |
| Built-in tools | `web_search`, `code_interpreter`, `mcp` | Not available (use function calling) |
| Per-message ID | `output.id` | Only top-level `response.id` |
| Reasoning | Dedicated `reasoning` items | `reasoning_content` field (when provider supplies it) |
| SGLang/vLLM compatibility | Partial (Responses not supported) | Full |

---

## Security Notes

- `openai_api_key` is never logged. The handler runs a regex-based redaction over error messages before logging them.
- Tool arguments and message bodies may contain user PII. The default `INFO` log line includes only metadata (model, request id, tokens, latency, finish reason); avoid raising the log level in production.
- `https:` image URLs are passed verbatim to the model provider, which fetches them. If your provider is in a trusted network, this expands its outbound surface — consider rewriting URLs through a content proxy you control if that matters.

---

## Production Checklist

Before deploying:

- Pin `openai` to the tested minor range (see `pyproject.toml`).
- Set `request_timeout_seconds`, `connect_timeout_seconds`, and pool limits explicitly for your traffic profile.
- Set `max_tool_call_depth` low enough to cap runaway spend but high enough for your longest legitimate tool chain.
- Confirm the logger level redacts secrets and keeps message bodies out of production logs.
- Wrap the handler in `with ... as handler:` or call `handler.close()` so the `httpx` pool is released.
- Run the deterministic test suite in your CI environment.

---

## Troubleshooting

- **`BadRequestError: messages must alternate ...`** — Your message history has an orphan `role: "tool"` message or an assistant `tool_calls` message without matching tool results. Check that your trimming logic preserves complete tool-call groups.
- **`Provider rejects parameter X`** — Some OpenAI-compatible servers do not support every standard parameter. Move provider-specific options into `extra_body` and remove them from the top-level configuration.
- **Stream ends with no content** — Some providers send a final usage-only chunk before terminating. The handler accounts for this; if you still see empty output, verify the model is returning text rather than only tool calls.
- **`ToolCallDepthExceeded`** — Either the model is in a tool-call loop or `max_tool_call_depth` is too low. Inspect the tool call sequence and raise the cap deliberately rather than as a workaround.

---

## License

MIT License - Copyright (c) 2025 IdeaBosque
