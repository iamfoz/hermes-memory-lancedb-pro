# Hermes Agent integration

This guide covers installing `hermes-memory-lancedb-pro` as a Hermes Agent
memory provider, how the discovery shim works, upgrades, and troubleshooting.

## The two pieces of an install

A working install always has two parts:

1. **The package** — all the code (`lancedb`, `sentence-transformers`, and the
   provider itself). This must live in **Hermes' own Python environment**.
2. **The discovery shim** — a tiny directory at `~/.hermes/plugins/lancedb_pro/`
   that lets hermes-agent find the provider.

## Step 1 — install the package

Install into Hermes' environment with `hermes-pip`:

```bash
hermes-pip install hermes-memory-lancedb-pro
```

Or from a clone:

```bash
hermes-pip install -e .
```

> **Use `hermes-pip`, not plain `pip`.** A plain `pip install` puts the package
> in whatever Python is currently active. hermes-agent loads the discovery shim
> inside *its own* environment; if the package is not there, the shim's `import`
> fails silently and the plugin never loads. `hermes-pip` is the wrapper that
> targets the correct environment.

## Step 2 — create the discovery shim

```bash
hermes-memory-lancedb-pro install-plugin
```

This creates `~/.hermes/plugins/lancedb_pro/` containing three files:

| File | Purpose |
|---|---|
| `__init__.py` | The discovery shim — re-exports `register` / `register_memory_provider` from the package. Those names double as the markers the host uses to auto-detect a memory provider. |
| `cli.py` | Re-exports `register_cli`, so `hermes lancedb_pro <command>` works. |
| `plugin.yaml` | Copied from the wheel — declares `name` and the hook list. It deliberately has **no `kind:`** — see below. |

The shim is just re-exports, so code upgrades never need the plugin directory
touched — `hermes-pip install -U hermes-memory-lancedb-pro` upgrades the
provider. The one exception is when the bundled `plugin.yaml` itself changes
(as in the release that dropped `kind: memory`): re-run
`hermes-memory-lancedb-pro install-plugin` afterwards. It detects the stale
copy and refreshes it automatically.

### Why the manifest has no `kind`

hermes-agent's manifest kinds are `standalone`, `backend`, `exclusive`,
`platform`, and `model-provider`. Memory providers are not declared — the host
auto-detects them from the `__init__.py` markers and routes them to the memory
discovery path (as `exclusive`). That auto-detection **only runs when `kind` is
absent**: an unrecognised value such as the old `kind: memory` is coerced to
`standalone`, skips detection, and the general plugin manager then eagerly
imports the plugin as a standalone plugin. Leave the field out.

`install-plugin` accepts:

```bash
hermes-memory-lancedb-pro install-plugin --hermes-home /path/to/profile  # non-default profile
hermes-memory-lancedb-pro install-plugin --force                         # overwrite an existing shim
```

It also auto-migrates older, incorrectly-placed installs and refreshes a stale
shim, so re-running it is always safe.

## Step 3 — activate the provider

In hermes-agent's `config.yaml`:

```yaml
memory:
  provider: lancedb_pro
```

Restart the gateway. Everything else — embedding-model warmup, the smart
extractor, admission control, reflection capture, automatic compaction — is
wired automatically.

## Entry-point discovery (no shim)

The package also registers a setuptools entry point:

```toml
[project.entry-points."hermes.plugins"]
lancedb_pro = "hermes_memory_lancedb_pro.provider:register"
```

hermes-agent builds that support `importlib.metadata` entry-point discovery find
the provider from the installed package alone — Step 2 is not needed. The shim
is kept for hosts that only scan `~/.hermes/plugins/`. The same package supports
both discovery paths.

## Uninstalling

```bash
hermes-memory-lancedb-pro uninstall-plugin
```

This removes only the files the installer created (`__init__.py`, `cli.py`,
`plugin.yaml`, `__pycache__`). If the directory contains other, unmanaged files
it is left in place.

## Troubleshooting

**The plugin does not load.** Confirm the package is importable from *Hermes'*
Python, not yours:

```bash
hermes-pip run python -c "import hermes_memory_lancedb_pro as m; print(m.__version__, m.__file__)"
```

**A stale copy is shadowing the install.** If `__file__` above points inside
`~/.hermes/plugins/lancedb_pro/` and the version is old, an old full checkout is
shadowing the installed package. Earlier guidance told users to clone the repo
*into* the plugin directory; that is no longer correct. Replace it with the shim:

```bash
mv ~/.hermes/plugins/lancedb_pro ~/.hermes/plugins/lancedb_pro.old   # back up
hermes-pip install -U hermes-memory-lancedb-pro
hermes-memory-lancedb-pro install-plugin
```

**The provider is loaded as a standalone plugin / not routed to the memory
manager.** `plugin.yaml` must **not** declare a `kind`. Older releases shipped
`kind: memory`, which current hermes-agent treats as invalid and coerces to
`standalone`. Check `~/.hermes/plugins/lancedb_pro/plugin.yaml`; if it still
has a `kind:` line, upgrade and re-run `hermes-memory-lancedb-pro install-plugin`
(it refreshes stale copies), or simply delete the line.

**Check store health at any time:**

```bash
hermes-memory-lancedb-pro doctor
```

## Related documentation

- [configuration.md](configuration.md) — every environment variable.
- [hooks.md](hooks.md) — the lifecycle hooks the provider implements.
- [jmunch.md](jmunch.md) — optional jmunch gateway support.
