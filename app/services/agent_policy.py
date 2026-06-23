"""Phase C — the LLM agent as a pluggable trading policy.

The agent plugs into the SAME walk-forward engine as every other policy
(`app/services/evaluation.py`). It is just a `callable[[PolicyContext], float]`
that returns a target long/flat exposure in [0, 1]. It only ever sees a
`PolicyContext` (today's regime, the model's forecast of tomorrow, and its own
previous position) — the engine forbids lookahead, so the agent cannot cheat.

Two practical guarantees:

  * **Reproducible & cheap.** Responses are cached by a hash of the exact prompt.
    The forecast is bucketed coarsely (0.05) before it reaches the model, so over
    a whole test window the agent faces only a few dozen UNIQUE contexts — not one
    call per day. The cache persists to disk: a re-run is free and byte-identical.

  * **Provider-agnostic.** Anthropic (hosted) or Ollama (local), selected by
    config. API keys come from the environment only; nothing is written in code.

SIMULATION ONLY. The agent decides a *paper* exposure — no order ever leaves the
process, no wallet, no key, no real funds.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.services.regime import STATE_ORDER

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/agent_cache")

SYSTEM_PROMPT = (
    "You are a disciplined paper-trading decision engine managing ONE long/flat "
    "position on a single asset. Given today's market regime and a model's forecast "
    "of tomorrow's regime, decide today's target exposure.\n"
    'Reply with raw JSON ONLY — no markdown, no prose: '
    '{"position": <number 0.0-1.0>, "reasoning": "<short note>"}\n'
    "position is the fraction of capital held long (1.0 = fully long, 0.0 = flat). "
    "Be cautious in bear or uncertain regimes. Every change in position pays a "
    "trading cost, so avoid needless churn."
)


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
class LLMProvider:
    """Minimal sync text-completion interface."""

    id = "base"
    model = ""

    def complete(self, system: str, user: str) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def health(self) -> bool:
        """Cheap reachability probe; returns False instead of raising."""
        try:
            self.complete("Reply with the single JSON object {\"ok\": 1}.", "{}")
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("%s provider health check failed: %s", self.id, e)
            return False


class AnthropicProvider(LLMProvider):
    id = "anthropic"

    def __init__(self, model: str, api_key: str, temperature: float = 0.0):
        self.model = model
        self._api_key = api_key
        self.temperature = temperature
        self._client = None

    def complete(self, system: str, user: str) -> str:
        import anthropic

        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self._api_key)
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=200,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text


class OllamaProvider(LLMProvider):
    id = "ollama"

    def __init__(self, model: str, host: str = "http://localhost:11434", temperature: float = 0.0):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature

    def complete(self, system: str, user: str) -> str:
        import httpx

        resp = httpx.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "format": "json",
                "options": {"temperature": self.temperature},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def make_provider(settings) -> LLMProvider | None:
    """Build the configured provider, or None if disabled/unconfigured."""
    name = (getattr(settings, "llm_provider", "none") or "none").lower()
    temp = float(getattr(settings, "agent_temperature", 0.0))
    if name == "anthropic":
        key = getattr(settings, "anthropic_api_key", "") or ""
        if not key:
            logger.warning("llm_provider=anthropic but ANTHROPIC_API_KEY is empty; agent disabled")
            return None
        return AnthropicProvider(getattr(settings, "anthropic_model", "claude-sonnet-4-20250514"), key, temp)
    if name == "ollama":
        return OllamaProvider(
            getattr(settings, "ollama_model", "qwen2.5:7b"),
            getattr(settings, "ollama_host", "http://localhost:11434"),
            temp,
        )
    if name not in ("none", ""):
        logger.warning("unknown llm_provider=%r; agent disabled", name)
    return None


# --------------------------------------------------------------------------- #
# Reproducible response cache
# --------------------------------------------------------------------------- #
class ResponseCache:
    """Prompt-keyed cache. Optionally persisted to a JSON file on disk.

    Contamination invariant: this is a PURE map from a hash of the prompt to the
    reply, and nothing else. The key depends only on prompt *content*
    (`_format_context` — today's regime + bucketed forecast + bucketed position),
    never on `day_index`, a timestamp, or anything derived from other days. That
    is what makes it the only sound form of "agentic memory" in a no-lookahead
    evaluation: it can never let day t see anything from day t±k. Do not turn it
    into an accumulating / similarity-retrieval store — `test_agent_invariants`
    guards the day_index half of this.
    """

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else None
        self._d: dict[str, str] = {}
        self.hits = 0
        self.misses = 0
        if self.path and self.path.exists():
            try:
                self._d = json.loads(self.path.read_text())
            except Exception:  # noqa: BLE001 - corrupt cache shouldn't kill a run
                self._d = {}

    def get_or_call(self, key: str, fn) -> str:
        if key in self._d:
            self.hits += 1
            return self._d[key]
        self.misses += 1
        val = fn()
        self._d[key] = val
        return val

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._d, indent=2, sort_keys=True))


# --------------------------------------------------------------------------- #
# Prompt building + parsing
# --------------------------------------------------------------------------- #
def _bucket(x: float, step: float) -> float:
    return round(round(x / step) * step, 4)


def _format_context(ctx) -> dict:
    """Compact, decision-relevant view of the context — coarsely bucketed so that
    near-identical days collapse to the same cached call."""
    p = np.asarray(ctx.p_next, dtype=float).ravel()
    m = min(len(STATE_ORDER), len(p))
    forecast = {STATE_ORDER[i]: _bucket(float(p[i]), 0.05) for i in range(m)}
    nxt = STATE_ORDER[int(np.argmax(p[:m]))] if m else "unknown"
    return {
        "today_regime": ctx.state,
        "forecast_next_regime": forecast,
        "most_likely_next": nxt,
        "current_position": _bucket(float(ctx.prev_pos), 0.1),
    }


_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_POS_KEYS = ("position", "exposure", "target", "target_position", "weight", "allocation")
_LABELED = re.compile(
    r"(?:position|exposure|target|weight|allocation)\s*(?:to|of|at|[:=])?\s*(-?\d+(?:\.\d+)?)",
    re.I,
)


def _parse_position(raw: str, fallback: float) -> tuple[float, str]:
    """Extract a target position in [0, 1] from a model reply, conservatively.

    Returns `(position, tier)` where `tier` is a STABLE tag recording HOW the
    number was recovered — `strict_json`, `salvaged_labeled`,
    `salvaged_single_number`, or `unparseable_hold`. The tier is the honest
    confidence signal: a run dominated by `strict_json` trusted the model
    cleanly; one full of salvage/hold did not. Callers count it (see AgentStats).

    Order of attempts: (1) strict JSON with a recognized key; (2) a labeled value
    in prose ("position: 0.7", "exposure to 0.8"); (3) salvage a bare number ONLY
    when the reply contains exactly one in-range number — so a stray number that
    precedes the real one (a z-score, a year, a window length) can't be mistaken
    for the position. Anything ambiguous holds the previous position.
    """
    text = (raw or "").strip()
    # 1) strict JSON with a recognized key
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group(0) if m else text)
        if isinstance(data, dict):
            for key in _POS_KEYS:
                if data.get(key) is not None:
                    return max(0.0, min(1.0, float(data[key]))), "strict_json"
    except Exception:  # noqa: BLE001
        pass
    # 2) a labeled position somewhere in prose
    lm = _LABELED.search(text)
    if lm:
        return max(0.0, min(1.0, float(lm.group(1)))), "salvaged_labeled"
    # 3) exactly one in-range number -> safe to use; multiple -> too ambiguous
    in_range = [v for v in (float(x) for x in _NUM.findall(text)) if 0.0 <= v <= 1.0]
    if len(in_range) == 1:
        return in_range[0], "salvaged_single_number"
    # 4) ambiguous or empty -> hold
    return float(fallback), "unparseable_hold"


# --------------------------------------------------------------------------- #
# Policy factory
# --------------------------------------------------------------------------- #
@dataclass
class AgentStats:
    """Per-decision provenance — how each agent position was obtained.

    This is the project's honesty idiom applied to the agent itself: instead of
    a self-asserted accuracy number, we report a MEASURED histogram of where the
    decisions came from. `strict_json` = the model answered cleanly;
    `salvaged_*` = we had to recover the number from messy prose; the two `*hold`
    counters = we got nothing usable and held the prior position. A run that is
    mostly salvage/hold is one whose agent row should be read with suspicion.
    """

    errors: int = 0                  # provider call raised -> held prev position
    strict_json: int = 0             # clean JSON with a recognized key
    salvaged_labeled: int = 0        # labeled value recovered from prose
    salvaged_single_number: int = 0  # lone in-range number recovered
    unparseable_hold: int = 0        # nothing usable -> held prev position

    def tally(self, tier: str) -> None:
        if hasattr(self, tier):
            setattr(self, tier, getattr(self, tier) + 1)

    @property
    def decisions(self) -> int:
        """Total agent decisions observed (every parse tier + held-on-error)."""
        return (self.strict_json + self.salvaged_labeled
                + self.salvaged_single_number + self.unparseable_hold + self.errors)


def build_agent_policy(provider: LLMProvider, cache: ResponseCache | None = None,
                       system: str = SYSTEM_PROMPT):
    """Return a `fn(PolicyContext) -> float` that asks `provider` for a target
    exposure. Identical (bucketed) contexts are served from `cache`. Any provider
    error or unparseable reply falls back to holding the previous position, so a
    single bad call never corrupts the whole walk-forward."""
    cache = cache if cache is not None else ResponseCache()
    stats = AgentStats()

    def policy(ctx) -> float:
        user = json.dumps(_format_context(ctx), sort_keys=True)
        key = hashlib.sha256(
            f"{provider.id}|{provider.model}|{system}|{user}".encode()
        ).hexdigest()
        try:
            raw = cache.get_or_call(key, lambda: provider.complete(system, user))
        except Exception as e:  # noqa: BLE001
            stats.errors += 1
            logger.warning("agent provider error -> holding (%s)", e)
            return float(ctx.prev_pos)
        pos, tier = _parse_position(raw, float(ctx.prev_pos))
        stats.tally(tier)
        return pos

    policy.stats = stats   # type: ignore[attr-defined]
    policy.cache = cache   # type: ignore[attr-defined]
    return policy
