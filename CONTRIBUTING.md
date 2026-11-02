# Contributing

This repository is an evaluation harness whose main claim is that its numbers reproduce.
Contributions are welcome when they keep that claim true. The rules below are the ones the
tests already enforce; this page just says them in one place.

## Set up

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env     # every default works offline; no key is needed
make test                # the whole suite runs without network access
```

Python 3.11 or newer. The pinned snapshots in `data/snapshots/` are the only data the tests
touch; nothing in the suite may call yfinance or an LLM provider.

## What a good change looks like

- **Causal.** A decision at day *t* uses data up to and including *t*, never later. If your
  change touches the walk-forward, say in the PR how you convinced yourself there is no lookahead.
- **Measured against the persistence bar.** New accuracy numbers are reported next to
  "tomorrow's regime = today's", not in isolation.
- **Honesty gates only append.** The warnings the engine emits about its own output are
  the product. A change may add a gate or sharpen its wording; it may not silence one.
- **Reproducible.** A new study pins its data (`load_or_fetch`) and writes a run record
  under `results/`. A changed headline number is a finding that belongs in the PR body and
  in `CHANGELOG.md`, not a silent diff in a JSON file.
- **Simulation only.** No broker, wallet, private key, on-chain call, or exchange API.
  Credentials come from `.env` and are never function arguments
  (`tests/test_agent_invariants.py` checks this).

## Tests

Put new tests in `tests/`, one file per concern, named `test_<area>_<concern>.py`. Prefer a
test that pins a boundary (what happens at exactly the threshold) over one that re-runs a
whole study. Run the suite unpiped so a failure stops you:

```bash
.venv/bin/python -m pytest -q
```

## Branches and commits

Branch prefixes in use: `fix/`, `perf/`, `chore/`, `test/`, `docs/`, `feat/`. Commit titles
are imperative sentences that say what changed and why it matters
("Flatten on non-finite strategy signals"), not labels ("update evaluation.py").

Every pull request gets an automated review within a few minutes of opening. Read its
inline comments, fix what is real, reply on the thread with the fixing commit, and only
then ask for a human review.
