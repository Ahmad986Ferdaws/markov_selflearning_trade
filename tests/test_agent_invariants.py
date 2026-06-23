"""Phase C — enforced invariants that keep the agent honest and safe *by
construction*, not by accident.

Today these guarantees hold because nobody has broken them yet (a grep
confirms it). These tests turn them into executable tripwires so a future
change can't quietly cross a line the project promised never to cross:

  1. the agent stays CLI-only — never reachable from the web/API surface, so it
     can never be publicly triggered on the user's Anthropic key;
  2. API keys come from settings/.env only — never passed as a call argument;
  3. the agent's decision is invariant to `day_index` — no position in the test
     timeline can leak into the prompt-hash cache. This is the no-lookahead
     guarantee made concrete: the article's "agentic memory" stays a pure
     prompt->reply map and never becomes an accumulating store.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

import app
from app.services.agent_policy import ResponseCache, build_agent_policy, make_provider
from app.services.evaluation import PolicyContext


class _FakeProvider:
    id, model = "fake", "fake-1"

    def __init__(self, reply: str = '{"position": 0.6}'):
        self.reply = reply
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return self.reply


# 1) CLI-only: the web surface must never import the agent --------------------
def test_agent_policy_stays_off_the_web_surface():
    app_dir = Path(app.__file__).parent
    web_files = [app_dir / "main.py"]
    api_dir = app_dir / "api"
    if api_dir.exists():
        web_files += list(api_dir.rglob("*.py"))
    offenders = [
        str(f.relative_to(app_dir.parent))
        for f in web_files
        if f.exists() and "agent_policy" in f.read_text()
    ]
    assert not offenders, (
        "The LLM agent must stay CLI-only (it must NOT be publicly triggerable on "
        f"the user's key). These web/API modules import agent_policy: {offenders}"
    )


# 2) env-only keys: no credential is ever a function argument -----------------
def test_keys_are_env_only_never_arguments():
    for fn in (make_provider, build_agent_policy):
        params = list(inspect.signature(fn).parameters)
        leaks = [
            p for p in params
            if any(w in p.lower() for w in ("key", "token", "secret", "password"))
        ]
        assert not leaks, (
            f"{fn.__name__} must take credentials from settings/.env only, never as "
            f"a call argument; found parameter(s): {leaks}"
        )


# 3) no-lookahead: the decision is invariant to day_index ---------------------
def test_agent_output_invariant_to_day_index():
    p = _FakeProvider('{"position": 0.6}')
    fn = build_agent_policy(p, ResponseCache())
    base = dict(state="bull", prev_pos=0.0, p_next=np.array([0.6, 0.2, 0.2]))
    early = fn(PolicyContext(**base, day_index=0))
    late = fn(PolicyContext(**base, day_index=999))
    assert early == late == 0.6
    # day_index never entered the cache key -> still exactly one provider call.
    # If it ever leaks in, this becomes 2 and the no-lookahead guarantee is gone.
    assert p.calls == 1
