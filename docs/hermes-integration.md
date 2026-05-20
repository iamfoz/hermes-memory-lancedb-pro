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
| `__init__.py` | The discovery shim — re-exports `register` / `register_memory_provider` from the package. |
| `cli.py` | Re-exports `register_cli`, so `hermes lancedb_pro <command>` works. |
| `plugin.yaml` | Copied from the wheel — declares `name`, `kind: memory`, and the hook list. |

The shim is just re-exports, so **upgrades never touch the plugin directory** —
`hermes-pip install -U hermes-memory-lancedb-pro` is the whole upgrade.

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

**The provider is found but not routed.** `plugin.yaml` must declare
`kind: memory`. The bundled file already does; if you hand-edited it, restore
that line so the host routes the plugin to the memory manager.

**Check store health at any time:**

```bash
hermes-memory-lancedb-pro doctor
```

## Related documentation

- [configuration.md](configuration.md) — every environment variable.
- [hooks.md](hooks.md) — the lifecycle hooks the provider implements.
- [jmunch.md](jmunch.md) — optional jmunch gateway support.
