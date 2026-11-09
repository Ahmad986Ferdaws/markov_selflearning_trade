# tests/ — the conventions the files already follow

The suite is the executable half of CONTRIBUTING.md. Everything below is already true of
the files in this folder; this page only writes it down.

## Shape

- One file per concern, named `test_<area>_<concern>.py`, opening with a module docstring
  that says what the file pins and why. Plain `def test_...` functions with descriptive
  names; async tests are plain `async def` because `asyncio_mode = auto` is set in
  `pyproject.toml`. There is no `conftest.py`.
- Prefer a boundary pin (what happens at exactly the threshold) over re-running a study.
  `pytest.approx`, `pytest.raises(..., match=...)`, `monkeypatch`, `tmp_path` and `caplog`
  are the tools in use.
- Fully offline. The LLM provider is a fake (`FakeProvider` in `test_agent_guards.py`,
  `_FakeProvider` in `test_agent_invariants.py`); `fetch_daily_history` is monkeypatched
  wherever data would be pulled (`test_data_cache_pinning.py`); everything else reads the
  pinned pickles in `data/snapshots/`. A test that reaches yfinance or a provider is a bug.

## The tripwire families

| Family | File | What it refuses to let change silently |
|---|---|---|
| Scope | [`test_scope_tripwires.py`](test_scope_tripwires.py) | No exchange, wallet or chain SDK imported anywhere; no credential-shaped literal committed; every `docs/` link in README resolves. |
| Web surface | [`test_web_surface.py`](test_web_surface.py) | The OpenAPI inventory is exactly the legacy `/runs` routes plus `/health`. |
| Agent invariants | [`test_agent_invariants.py`](test_agent_invariants.py) | The agent stays CLI-only and credentials never travel as function arguments. |
| Version | [`test_version_consistency.py`](test_version_consistency.py) | `app.__version__`, `pyproject.toml` and the newest CHANGELOG heading agree. |
| Committed evidence | [`test_committed_results_records.py`](test_committed_results_records.py), [`test_instrument_validation.py`](test_instrument_validation.py) | The run records say what README and docs/16, 18 and 19 quote; the harness detects planted structure (positive controls) and reproduces the zero on sticky synthetic data. |

## API tests

Routes are exercised through FastAPI's `TestClient` against an in-memory SQLite engine built
with `StaticPool` and `check_same_thread=False`: the client serves requests on a worker
thread, so one shared connection is needed. The app's `get_db_dep` and `get_settings_dep`
are replaced through `app.dependency_overrides`, so each test chooses its `Settings`
explicitly. See [`test_web_surface.py`](test_web_surface.py) and
[`test_legacy_api_guards.py`](test_legacy_api_guards.py).

## Running

```bash
.venv/bin/python -m pytest -q                               # everything
.venv/bin/python -m pytest -q tests/test_guards.py          # one file
.venv/bin/python -m pytest -q -W error::ResourceWarning     # an unclosed file or socket fails the test
```

Run pytest unpiped. In a pipeline such as `pytest | tail` the exit code is `tail`'s, so a
failing suite looks green, and `set -o pipefail` has not reliably caught it either. If you
must pipe, capture the return code first and gate on it:

```bash
.venv/bin/python -m pytest -q > /tmp/pytest.out 2>&1; rc=$?; tail -3 /tmp/pytest.out; echo rc=$rc
```
