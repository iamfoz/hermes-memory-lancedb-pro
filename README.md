# hermes-memory-lancedb-pro

[![Standard README compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg)](https://github.com/RichardLitt/standard-readme)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

> LanceDB-backed persistent memory for [Hermes Agent](https://github.com/nousresearch/hermes-agent) — hybrid BM25 + vector search with Weibull decay, tiered retention, and LLM-driven extraction.

`hermes-memory-lancedb-pro` is a drop-in **memory provider plugin** for Hermes
Agent. It gives an agent durable, searchable recall across sessions: every
conversation turn is distilled into structured memories, scored for relevance
and freshness, and surfaced back into the prompt when it matters — without the
"sticky memory" bleed that makes naive memory stores worse than none.

It also works as a standalone Python library, with no dependency on
hermes-agent, for any project that needs a hybrid-search memory store.

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
  - [Using it in Hermes Agent](#using-it-in-hermes-agent)
  - [Using it as a standalone library](#using-it-as-a-standalone-library)
- [Configuration](#configuration)
- [jmunch gateway support](#jmunch-gateway-support)
- [Hermes hooks](#hermes-hooks)
- [Documentation](#documentation)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Background

An LLM agent with a memory store fails in a predictable way: old memories from
unrelated tasks keep getting injected into fresh conversations, the model
conflates them with the current question, and recall quality collapses. This
project is built around *not* doing that.

Core capabilities:

- **Hybrid search** — Reciprocal Rank Fusion of BM25 (lexical) and cosine
  (semantic, `nomic-embed-text-v1.5`, 768-d) retrieval.
- **Weibull decay** — `recency = exp(-(λ·t)^β)` with a per-tier `β`
  (core 0.8 / working 1.0 / peripheral 1.3); recency is exactly 0.5 at each
  tier's half-life.
- **Tiered retention** — core / working / peripheral tiers with automatic
  promotion and demotion.
- **Session scoping** — recall is restricted to the current session unless a
  memory is explicitly cross-session or core-tier, which is what stops the
  stickiness bleed.
- **LLM-driven extraction** — an optional smart extractor distils turns into a
  six-category schema with admission control and deduplication.
- **Durable task ledger** — multi-step task state persisted to disk so it
  survives context compaction.
- **Supersede pattern** — updates archive the old row and write a new one,
  preserving a full audit trail with no vector drift.

See [docs/architecture.md](docs/architecture.md) for the full design.

## Install

**Requirements:** Python 3.11+ and a working Hermes Agent installation.

Install the package into **Hermes' own Python environment** with `hermes-pip`,
then create the discovery shim:

```bash
# 1. Install into Hermes' environment (not your system Python)
hermes-pip install hermes-memory-lancedb-pro

# 2. Create the plugin discovery shim under ~/.hermes/plugins/lancedb_pro/
hermes-memory-lancedb-pro install-plugin
```

> **Why `hermes-pip`?** This is the single most common install mistake. A plain
> `pip install` lands the package in whatever Python is active, but hermes-agent
> loads the discovery shim inside *its own* environment. If the package isn't
> there, the import fails silently and the plugin never loads. `hermes-pip`
> targets the correct environment.

A working install has two pieces: the **package** holds all the code, and the
**shim** at `~/.hermes/plugins/lancedb_pro/` is ~5 lines that let hermes-agent
discover the provider. Because the shim only re-exports the package, upgrades
are just `hermes-pip install -U hermes-memory-lancedb-pro` — no need to touch
the plugin directory.

Newer hermes-agent builds that support entry-point discovery
(`importlib.metadata`, group `hermes.plugins`) find the provider without the
shim — step 1 alone is enough.

Full install detail, upgrade notes, and troubleshooting are in
[docs/hermes-integration.md](docs/hermes-integration.md).

## Usage

### Using it in Hermes Agent

After [installing](#install), activate the provider in hermes-agent's
`config.yaml`:

```yaml
memory:
  provider: lancedb_pro
```

Restart the gateway. That is the entire setup — everything else (embedding-model
warmup, the smart extractor, admission control, reflection capture, automatic
compaction) is wired automatically, and each piece has an environment-variable
off switch if you need it.

The provider exposes an admin CLI for inspecting and maintaining the store:

```bash
hermes-memory-lancedb-pro doctor          # health report + recommendations
hermes-memory-lancedb-pro export -o backup.jsonl
hermes-memory-lancedb-pro import --in backup.jsonl --reembed
```

When the plugin is active, the same commands are also reachable through
hermes-agent's CLI as `hermes lancedb_pro <command>`.

### Using it as a standalone library

The store, retriever, decay engine, and extraction pipeline import and run
**without hermes-agent installed**:

```python
from hermes_memory_lancedb_pro import MemoryStore, MemoryRetriever

store = MemoryStore.get_instance()          # path-keyed singleton

store.store(
    text="Martyn prefers concise responses in UK English.",
    category="preference", scope="global", importance=0.9,
)

retriever = MemoryRetriever(store)
hits = retriever.retrieve("how should I reply?", limit=5, session_id="sess-1")
for h in hits:
    print(h["score"], h["text"])
```

More library examples — bulk writes, the smart extractor, the reflection layer,
the memory compactor — are in [docs/usage.md](docs/usage.md).

## Configuration

Every tuning knob is an environment variable, and all of them are optional —
the defaults are production-ready. The most common ones:

| Variable | Default | Purpose |
|---|---|---|
| `MEMORY_DB_DIR` | `~/.hermes/memory-lancedb` | Database directory |
| `MEMORY_PREFETCH_LIMIT` | `5` | Memories recalled per turn |
| `MEMORY_MIN_RECALL_SCORE` | `0.0` | Score floor for recall (raise to ~0.2 to drop weak matches) |
| `MEMORY_ADMISSION_PRESET` | `balanced` | Admission gate: `balanced` / `conservative` / `high-recall` / `off` |
| `MEMORY_REFLECTION` | `on` | Capture and replay session reflections |

The **complete reference** — every variable, default, and grouping — is in
[docs/configuration.md](docs/configuration.md).

## jmunch gateway support

[jmunch](https://github.com/iamfoz/jmunch-mcp) is an optional gateway that sits
between the agent and its LLM and reduces token cost by "handle-ifying" large
tool results. That lossily compresses the agent's conversation history — so when
a jmunch gateway is present, this plugin **compensates**: it widens recall and
loosens the admission gate to push more task context back through the memory
block (which is never handle-ified), and it asks the gateway to relay
memory-extraction calls verbatim so the extractor sees full-fidelity content.

This is entirely optional and has **no code dependency** on jmunch — the two
projects are orthogonal. Detection is automatic (via the `X-Jmunch-Gateway`
response header) and can be declared up front with `MEMORY_JMUNCH_MODE=true`.
The gateway-side support it relies on currently lives in a fork of jmunch-mcp,
not yet an upstream release.

See [docs/jmunch.md](docs/jmunch.md) for the full integration guide.

## Hermes hooks

The provider implements the standard Hermes memory-provider lifecycle hooks
(`system_prompt_block`, `prefetch`, `sync_turn`, `on_pre_compress`,
`on_memory_write`, `on_session_switch`, `on_session_end`, `shutdown`).

It *also* opportunistically implements two **non-standard** hooks
(`on_recall_used`, `on_tool_call_observed`) introduced on a parallel
hermes-agent branch. On a host that has them they sharpen recall-credit
accounting and context capture; on a host that doesn't, they are simply never
called — the same plugin works against both.

The full hook-by-hook reference is in [docs/hooks.md](docs/hooks.md).

## Documentation

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Storage model, retrieval pipeline, design decisions |
| [docs/hermes-integration.md](docs/hermes-integration.md) | Install, the discovery shim, upgrades, troubleshooting |
| [docs/configuration.md](docs/configuration.md) | Complete environment-variable reference |
| [docs/usage.md](docs/usage.md) | Standalone-library recipes and the CLI reference |
| [docs/hooks.md](docs/hooks.md) | Every memory-provider hook, standard and non-standard |
| [docs/jmunch.md](docs/jmunch.md) | jmunch gateway detection and compensation |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

## Maintainers

[@iamfoz](https://github.com/iamfoz) (Martyn Forryan).

## Contributing

Issues and pull requests are welcome. Please read
[CONTRIBUTING.md](CONTRIBUTING.md) first — it covers the development setup, the
test suite, and the coding conventions — and note the
[Code of Conduct](CODE_OF_CONDUCT.md). Security issues should follow the
process in [SECURITY.md](SECURITY.md) rather than a public issue.

This project follows the [Standard Readme](https://github.com/RichardLitt/standard-readme)
specification.

## License

[MIT](LICENSE) © Martyn Forryan.
