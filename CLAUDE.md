# CLAUDE.md

Guidance for AI coding assistants working in this repository. The conventions
below are durable user preferences and apply to **every** session — follow them
without needing to be reminded.

## Git attribution

- Do **not** add AI/Claude co-author trailers or any AI mention to commits,
  branch names, or pull requests. No `Co-Authored-By: Claude…`, no
  "Generated with…" footer, no session links.
- Author **and** commit everything as the user:
  `Martyn Forryan <iamfoz@users.noreply.github.com>`.
- Branch names use conventional prefixes — `feat/`, `fix/`, `chore/`, etc.
  Never use a `claude/` prefix.

## Git workflow

- Keep a clean, linear history.
- Force-pushing `main` is acceptable.
- Never change git config without an explicit request from the user.

## Environment

- The user develops on an Apple Silicon (ARM64) Mac. Prefer tooling, binaries,
  and instructions that work natively on `arm64-darwin`.
