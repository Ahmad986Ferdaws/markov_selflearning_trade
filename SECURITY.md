# Security policy

## Scope

This project is a simulation-only research harness. It holds no funds, places no orders,
and talks to no broker, wallet, or exchange. The things that *are* security-relevant here:

1. **Credential exposure.** An `ANTHROPIC_API_KEY` must only ever be read from `.env`
   (never committed; `.env` is git-ignored and `.env.example` ships empty values).
2. **A publicly triggerable agent.** The LLM agent is CLI-only by policy. Any path that
   lets an unauthenticated request spend the key (the legacy web API, a hosted demo) is
   a vulnerability, not a feature. `ALLOW_LEGACY_AGENT_API` defaults to off for this reason.
3. **Data integrity.** Pinned snapshots and run records are the evidence. A way to make a
   published number change without the data hash changing is a reportable bug.

## Reporting

Use GitHub's private vulnerability reporting on this repository (Security tab, "Report a
vulnerability") for anything in categories 1 or 2. For category 3, an ordinary issue with
the *Methodology concern* template is fine unless the report includes a key or a private URL.

Please include the command or request that demonstrates the problem. Expect an
acknowledgement within a week; this is a one-maintainer project.

## Dependencies

Dependabot raises the `>=` floors in `pyproject.toml` and the GitHub Actions pins once a
week (Monday). `requirements.lock` is the audited set: a reproduction installs from it, not
from the floors. A vulnerable pin is reported like category 3, as an ordinary issue, unless
exploiting it needs the API key; then use private reporting. A dependency bump cannot bring
an exchange or wallet SDK into the code path: `tests/test_scope_tripwires.py` fails on any
import of one in `app/`, `kalman/`, `scripts/` or `tests/`.

## Out of scope

- Findings that require a modified `.env` (for example `ALLOW_LEGACY_AGENT_API=true`) and
  local access to the machine running it.
- Third-party CDN scripts on the static site: they are pinned by Subresource Integrity
  hashes; a report that a pinned hash is wrong is in scope, a report about the CDN is not.
