# Hermes hooks

`LanceDBProMemoryProvider` is a Hermes Agent memory-provider plugin. Hermes
drives a memory provider through a set of **lifecycle hooks** — methods the host
calls at defined points in a session. This document lists every hook the
provider implements.

The hooks a provider offers are declared in `plugin.yaml`. A host only ever
calls hooks it knows about, so declaring a hook the host does not recognise is
harmless — it is simply never invoked.

## Standard hooks

These are part of the Hermes memory-provider contract; every supported
hermes-agent version drives them.

### `system_prompt_block`

Returns query-independent text for the **system prompt**: the durable-task
protocol and the current conversation's active-task control block. The block is
rendered from immutable fields where possible so it stays byte-stable across
turns, which preserves the host's prompt cache. The hook is guarded so it can
never raise into the host.

### `prefetch`

The query-dependent recall path — the hook the host calls every turn to pull
relevant memories. It runs the full `MemoryRetriever` pipeline, scoped to the
current session, and returns a formatted recall block (relevant memories plus
the reflection block). This is the only recall hook the host actually drives.

### `sync_turn`

Persists a completed conversation turn. The work runs in a **daemon thread** so
the hook returns immediately and never blocks the agent. When an extraction LLM
is configured the smart extractor distils the turn into structured memories;
otherwise the raw user turn is written directly.

### `on_pre_compress`

Called immediately before the host compresses context and discards old
messages. The provider ensures a session recovery anchor exists and returns the
current active-task control block so the host can fold it into the compression
summary — this is what lets a long task survive compaction.

### `on_memory_write`

Mirrors the host's built-in memory tool. When the user runs a built-in
`/memory add` or `/memory replace`, the same write is reflected into this store
so built-in memory and recall stay in sync. (`remove` is not mirrored.)

### `on_session_switch`

Handles a change of session. On a genuine reset it archives the conversation's
auto-anchors and advances the internal conversation id; on a non-reset switch it
keeps the conversation id. Stale pending recall-credit ids are dropped.

### `on_session_end`

Called when a conversation ends (not at process exit). It joins pending writes,
writes a session-summary memory, writes a reflection if an LLM is available,
flushes the recall-credit ledger, and triggers cooldown-gated auto-purge and
auto-compaction.

### `shutdown`

Called at process exit. It joins the pending `sync_turn` thread, clears internal
ledgers, and triggers cooldown-gated auto-purge and auto-compaction.

## Non-standard hooks

These two hooks come from a parallel hermes-agent branch
(`feat/memory-provider-hooks`). The provider implements them opportunistically:
on a host that has them they add value; on a host that does not, the host simply
never calls them and the plugin behaves identically. The **same wheel works
against both** host versions.

### `on_recall_used`

Credits the memories the assistant's response actually referenced, matched by
phrase overlap, and consumes the pending recall-credit ledger. This closes the
recall-frequency loop precisely: only memories the model *used* get their
access count bumped, instead of every memory that was merely fetched.

### `on_tool_call_observed`

Currently a no-op placeholder for future entity-extraction logic. It is wired so
that, on a host that emits the hook, tool-call context becomes available to the
extractor later without a plugin change.

## Deliberately not implemented: `before_prompt_build`

`before_prompt_build` is intentionally **not** implemented. Overriding it makes
the host's `prefetch_all()` skip this provider, and current hosts never call the
hook anyway. Recall is delivered through `prefetch` and `system_prompt_block`
instead. This is a deliberate choice, not an omission — see the comment in
`plugin.yaml`.

## Declared hook list

For reference, the `hooks:` list in `plugin.yaml`:

```yaml
hooks:
  - system_prompt_block
  - prefetch
  - sync_turn
  - on_pre_compress
  - on_memory_write
  - on_recall_used          # non-standard
  - on_tool_call_observed   # non-standard
  - on_session_switch
  - shutdown
  - on_session_end
```
