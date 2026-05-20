# Security Policy

## Supported versions

`hermes-memory-lancedb-pro` is an actively developed `0.y.z` project. Security
fixes are applied to the latest released minor version. Please upgrade to the
most recent release before reporting an issue.

| Version | Supported |
|---|---|
| 0.12.x | Yes |
| < 0.12 | No — please upgrade |

## Reporting a vulnerability

**Do not open a public issue for a security vulnerability.**

Report it privately through GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability):
go to the repository's **Security** tab and choose **Report a vulnerability**.
This opens a private advisory visible only to you and the maintainers.

Please include:

- the package version (`hermes-memory-lancedb-pro doctor` reports it);
- a description of the vulnerability and its impact;
- a minimal reproduction or proof of concept;
- any known mitigation.

## What to expect

- An acknowledgement within a few days of the report.
- An assessment of severity and affected versions.
- A fix developed privately, released as a new patch version, and disclosed in
  the [CHANGELOG.md](CHANGELOG.md) under a `Security` heading once users have had
  a reasonable window to upgrade.

We will credit reporters who wish to be named.

## Scope notes

This package handles untrusted text — conversation turns and tool output — and
takes some precautions accordingly:

- **Prompt-injection guard** — write-time text is screened by
  `MEMORY_INJECTION_GUARD` (`off` / `warn` / `reject` / `sanitize`). Reflection
  writes additionally pass `sanitize_injectable_reflection_lines`.
- **SQL-injection guards** — all store filter paths are parameterised or
  application-filtered; archived-row filtering is done in Python rather than
  through SQL `LIKE` patterns.

Reports that strengthen these paths, or that demonstrate a bypass, are
especially welcome.
