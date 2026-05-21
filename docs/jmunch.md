# jmunch gateway support

[jmunch](https://github.com/iamfoz/jmunch-mcp) is an optional gateway that sits
between an application and its upstream OpenAI- or Anthropic-compatible LLM API.
It reduces token cost by **handle-ifying** large tool results — replacing a fat
payload with a short summary plus an opaque handle.

This plugin supports jmunch but does **not** depend on it. The integration is
pure standard library; the two projects are completely orthogonal and can be
installed together or separately. If you do not use jmunch, nothing in this
document applies and nothing changes.

> **Gateway-side support.** The pieces jmunch itself must provide — stamping
> the `X-Jmunch-Gateway` header and honouring the `X-Jmunch-Inject` /
> `X-Jmunch-Handleify` request headers — currently live in a fork of jmunch-mcp
> and are not yet part of an upstream release. Against a stock jmunch that
> lacks them, this plugin simply never detects a gateway and behaves exactly as
> it does without jmunch.

## Why the plugin cares

Handle-ification is lossy: it compresses the agent's conversation history, so
over a long run the agent loses task detail that was in earlier tool results.

The memory block this plugin injects, however, is **not** handle-ified — it is a
lossless channel back into the prompt. So when a jmunch gateway is detected, the
plugin *compensates*:

- **Recall is widened** — a higher prefetch limit and a more permissive
  `min_score`, so more task context is re-surfaced through the lossless memory
  block.
- **Admission is loosened** — if the admission preset was not pinned explicitly,
  it is raised to `high-recall`, so fewer task-relevant candidates (notably
  progress events) are rejected before they are ever stored.
- **Extraction calls are passed through verbatim** — the memory extractor's own
  LLM calls ask the gateway not to inject or handle-ify, so the extractor sees
  full-fidelity tool content.

Recall configuration is re-evaluated on **every recall**, so a jmunch gateway
confirmed mid-session takes effect from the very next turn.

## How detection works

Detection is two-stage and needs no configuration:

1. **Startup declaration (optional).** Set `MEMORY_JMUNCH_MODE=true` and the
   plugin treats jmunch as in-use from process start — before the first LLM
   call. This is the only signal available ahead of the first response, so it
   is worth setting if you know a gateway is in the path: it makes turn-one
   tuning correct.

2. **Passive confirmation (automatic).** A pass-through-capable jmunch gateway
   stamps an `X-Jmunch-Gateway` header on every response. The plugin inspects
   response headers and latches the observation the first time it sees one,
   keying on the header's *presence* rather than any value it carries. This
   works on any port and needs no setup, but only takes effect from the first
   response onward.

`is_jmunch_in_use()` is true once **either** signal has fired.

## Headers

| Header | Direction | Meaning |
|---|---|---|
| `X-Jmunch-Gateway` | response | Stamped by a pass-through-capable jmunch gateway. Its presence is the passive detection signal; the value is not interpreted. |
| `X-Jmunch-Inject: false` | request | Asks the gateway not to inject verbs into this call. |
| `X-Jmunch-Handleify: false` | request | Asks the gateway not to handle-ify tool content, so the extractor sees raw payloads. |

The two request headers are inert on any non-jmunch endpoint, and a jmunch
gateway that does not recognise them simply ignores them.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MEMORY_JMUNCH_MODE` | *(none)* | Declare a jmunch gateway up front. Truthy: `1` / `true` / `yes` / `on`. |
| `MEMORY_JMUNCH_PREFETCH_LIMIT` | `12` | Recall limit while jmunch is in use (replaces `MEMORY_PREFETCH_LIMIT`). |
| `MEMORY_JMUNCH_MIN_RECALL_SCORE` | `0.0` | Recall score floor while jmunch is in use. An explicitly configured `min_score` is never overridden. |

A typical setup that routes the memory extractor through the same local jmunch
proxy as the agent:

```bash
# ~/.hermes/.env
MEMORY_EXTRACTION_PROVIDER=openai
MEMORY_EXTRACTION_BASE_URL=http://127.0.0.1:7879/v1
MEMORY_EXTRACTION_MODEL=Qwen3.6
MEMORY_EXTRACTION_API_KEY=local        # jmunch usually ignores the value
MEMORY_JMUNCH_MODE=true                # declare jmunch up front
```

## Public API

The `hermes_memory_lancedb_pro.jmunch` module is import-safe everywhere:

```python
from hermes_memory_lancedb_pro import (
    is_jmunch_in_use,        # True once either detection signal has fired
    jmunch_mode_configured,  # True only when MEMORY_JMUNCH_MODE is set
    jmunch_request_headers,  # headers to attach to an LLM call (empty off-jmunch)
    record_response_headers, # latch an X-Jmunch-Gateway observation
)
```

`jmunch_request_headers()` returns an empty dict when jmunch is not in use, so
callers can splat it onto a request unconditionally.
