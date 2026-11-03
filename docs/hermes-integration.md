# Hermes Agent integration

This guide covers installing `hermes-memory-lancedb-pro` as a Hermes Agent
memory provider, upgrades, and troubleshooting.

## Install

This repository is itself a Hermes plugin. Install it with Hermes' plugin
manager:

```bash
hermes plugins install iamfoz/hermes-memory-lancedb-pro
```

Accept the prompt to make it the active memory provider (or set
`memory.provider: lancedb_pro` in `config.yaml` yourself), then restart the
gateway. Verify with:

```bash
hermes memory status        # Provider: lancedb_pro / Plugin: installed ✓
hermes plugins doctor lancedb_pro
```

Everything else — embedding-model warmup, the smart extractor, admission
control, reflection capture, automatic compaction — is wired automatically.

### What happens under the hood

The repo root holds a `plugin.yaml` (`name: lancedb_pro`, so the plugin lands
in `~/.hermes/plugins/lancedb_pro/`), an `__init__.py` that re-exports the
provider's `register` / `register_memory_provider`, and a `cli.py` for
`hermes lancedb_pro <command>`.

Because the repo also has a `pyproject.toml` with a build backend, Hermes'
package manager (PM) treats the plugin as an installable project: it builds
the package **and its dependencies** (`lancedb`, `sentence-transformers`, …)
into Hermes' managed environment. PM includes the active memory provider's
requirements in every environment generation it builds, so `hermes update`
keeps them.

> **Do not use `pip` (or the old `hermes-pip`) against Hermes' environment.**
> Current Hermes rebuilds its environment from its own lock plus enabled
> plugins' declared dependencies. A package installed by hand is not
> declared, so the next environment generation drops it — and the plugin
> fails with `Memory provider 'lancedb_pro' loaded but no provider instance
> found` (the underlying `No module named 'hermes_memory_lancedb_pro'` is
> logged only at DEBUG).

### Why the manifest has no `kind`

hermes-agent's manifest kinds are `standalone`, `backend`, `exclusive`,
`platform`, and `model-provider`. Memory providers are not declared — the host
auto-detects them from the `__init__.py` markers and routes them to the memory
discovery path (as `exclusive`). That auto-detection **only runs when `kind` is
absent**: an unrecognised value such as the old `kind: memory` is coerced to
`standalone`, skips detection, and the general plugin manager then eagerly
imports the plugin as a standalone plugin. Leave the field out.

## Upgrading

```bash
hermes plugins update lancedb_pro
```

PM validates the new dependency set before publishing the new code.

## Migrating from a `hermes-pip` + `install-plugin` install

Older releases had you `hermes-pip install` the package and then run
`hermes-memory-lancedb-pro install-plugin` to write a small shim into
`~/.hermes/plugins/lancedb_pro/`. On current Hermes that package is not part of
PM's environment and disappears on update. Replace the shim with a plugin
install — your memory database (`~/.hermes/memory-lancedb/`) and task ledgers
are untouched:

```bash
mv ~/.hermes/plugins/lancedb_pro ~/.hermes/plugins/lancedb_pro.old   # the old shim
hermes plugins install iamfoz/hermes-memory-lancedb-pro
hermes memory status
```

The installer refuses to overwrite an existing `lancedb_pro` directory, which
is why the old shim has to move first. Delete the `.old` directory once
`hermes memory status` shows the plugin installed.

## Removing

```bash
hermes plugins remove lancedb_pro
```

This removes the plugin code. The memory database is left on disk.

## Standalone CLI and `install-plugin`

The package still installs the `hermes-memory-lancedb-pro` / `hermes-memory`
console scripts for using the store outside Hermes. Its `install-plugin` /
`uninstall-plugin` commands write and remove the legacy shim; they are only
useful on pre-PM Hermes builds where you manage Hermes' environment yourself.

## Entry-point discovery

For hosts that discover pip-installed memory providers, the package registers:

```toml
[project.entry-points."hermes_agent.memory_providers"]
lancedb_pro = "hermes_memory_lancedb_pro.provider:register"
```

It deliberately does **not** register under `hermes_agent.plugins`, the general
plugin group, which the general plugin manager imports eagerly.

## Troubleshooting

**`loaded but no provider instance found` / `hermes memory status` says
`NOT installed`.** The plugin directory was found, but its import of
`hermes_memory_lancedb_pro` failed — almost always because the package is not
in Hermes' environment. On an old shim install, follow
[Migrating](#migrating-from-a-hermes-pip--install-plugin-install). On a plugin
install, `hermes plugins doctor lancedb_pro` reports the failing check, and
`hermes pm install` re-syncs the environment.

**The provider is loaded as a standalone plugin / not routed to the memory
manager.** `plugin.yaml` must **not** declare a `kind`. Check
`~/.hermes/plugins/lancedb_pro/plugin.yaml`; if it has a `kind:` line, update
the plugin (`hermes plugins update lancedb_pro`) or delete the line.

**Check store health at any time:**

```bash
hermes lancedb_pro doctor
```

## Related documentation

- [configuration.md](configuration.md) — every environment variable.
- [hooks.md](hooks.md) — the lifecycle hooks the provider implements.
- [jmunch.md](jmunch.md) — optional jmunch gateway support.
