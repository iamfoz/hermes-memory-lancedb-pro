# Contributing

Thanks for considering a contribution to `hermes-memory-lancedb-pro`. This
document covers the development setup, the test suite, and the conventions the
project follows.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- **Report a bug** — open an issue with the [bug report](.github/ISSUE_TEMPLATE/bug_report.md)
  template. Include the package version (`hermes-memory-lancedb-pro doctor`) and
  a minimal reproduction.
- **Request a feature** — open an issue with the
  [feature request](.github/ISSUE_TEMPLATE/feature_request.md) template.
- **Send a pull request** — for anything beyond a trivial fix, open an issue
  first so the approach can be agreed before you invest time.
- **Report a security issue** — do **not** open a public issue; follow
  [SECURITY.md](SECURITY.md).

## Development setup

Requires Python 3.11 or newer.

```bash
git clone https://github.com/iamfoz/hermes-memory-lancedb-pro
cd hermes-memory-lancedb-pro
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

You do **not** need hermes-agent installed to develop or test the library —
the provider adapter imports it lazily.

## Running the tests

```bash
# Fast unit tests — no LanceDB or model download
pytest -m "not integration"

# Full suite, including LanceDB-backed tests (uses a stub embedder)
pytest
```

Every change should keep the full suite green. New behaviour needs new tests;
bug fixes need a regression test that fails before the fix.

## Linting

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and import
sorting. Run it before opening a pull request:

```bash
ruff check src/ tests/
ruff check --fix src/ tests/    # auto-fix import order and the like
```

CI-equivalent expectations: `ruff check` is clean and `pytest` passes.

## Coding conventions

- **Style** — follow the surrounding code. Ruff settings live in
  `pyproject.toml`; do not loosen them to pass a check.
- **Lazy heavy imports** — importing the package must not pull in `lancedb` or
  `sentence-transformers`. Keep heavy imports inside functions or behind the
  `__getattr__` lazy table in `__init__.py`.
- **Public API** — anything new that consumers should use goes in the package
  `__all__`.
- **Comments** — explain *why*, not *what*. Skip comments that merely restate
  the code.

## Commit and branch conventions

- Branch names are prefixed by intent: `feat/<topic>`, `fix/<topic>`,
  `docs/<topic>`, `chore/<topic>`.
- Write commit subjects in the imperative mood, prefixed by type — for example
  `fix: close the cold-start recall race` or `feat: add the durable task
  ledger`. Keep the subject under ~72 characters; put detail in the body.
- One logical change per commit where practical.

## Versioning and the changelog

- The project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
  While in `0.y.z`, a breaking change bumps the minor version.
- Every user-visible change gets an entry in [CHANGELOG.md](CHANGELOG.md) under
  the appropriate `Added` / `Changed` / `Deprecated` / `Removed` / `Fixed` /
  `Security` heading, following the
  [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.
- Do not bump the version string yourself in a feature PR — releases are cut by
  the maintainer.

## Pull request checklist

Before marking a PR ready for review:

- [ ] `pytest` passes and new behaviour is covered by tests.
- [ ] `ruff check src/ tests/` is clean.
- [ ] A `CHANGELOG.md` entry is added for any user-visible change.
- [ ] Documentation under `docs/` is updated if behaviour or configuration
      changed.
- [ ] The PR description explains the *why*, and links the issue it closes.
