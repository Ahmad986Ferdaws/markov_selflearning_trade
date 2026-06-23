# REGIME — the model has no edge

An honest, **simulation-only** daily evaluation harness for a Markov regime trading model.
It is not a profit bot and it is not self-learning. It exists to answer one question with data,
fairly, and to report the answer even when the answer is *no*.

> **The question:** does a Markov regime model actually predict the market better than the
> dumbest baseline in finance — *"tomorrow's regime = today's"*?
>
> **The answer:** no. The edge is **exactly zero**. This repo is the receipt.

---

## The receipt

On held-out BTC-USD (n = 328 days the model never saw during selection):

| Predictor | Hit-rate |
|---|---|
| **Markov model** | **90.9%** |
| **Persistence** ("tomorrow = today") | **90.9%** |
| Naive (always majority) | 72.0% |

The model looks impressive at 90.9% — until you notice that *guessing "same as today"* scores
**identically**. The transition matrix adds nothing over plain autocorrelation. Re-tested across
**20 markets** (crypto, equities, gold, bonds, oil, FX), the edge over persistence is **+0.000 on
20/20** — verified as a pointwise identity: **0 divergences across 5,425 predictions**. And net of
realistic costs (0.3% fee + 0.5% slippage), both trading policies lose: regime baseline **−41.1%**,
buy-and-hold **−47.7%**.

![The verdict — 90.9% ≡ 90.9%](media/02-verdict.png)
![Held-out equity, net of costs](media/04-results.png)

A ~21-second real-time walk-through of the site is in [`media/regime-demo.mp4`](media/regime-demo.mp4).

## Why the edge is zero

It's structural, not bad luck. The estimated transition matrix is **diagonal-dominant** —
self-transition probabilities dominate, so `argmax(P[today])` almost always lands back on today's
regime. That *is* the persistence rule. The model is an expensive way to rediscover autocorrelation.
Full derivation and the cross-asset reproduction in
[docs/16 — robustness study](docs/16-robustness-study.md).

## What this is

An **honest evaluation harness** that reports null when null is true
([docs/13 — reframed plan](docs/13-reframed-plan.md)). The design choices are all in service of
*not fooling yourself*:

- **No lookahead.** Walk-forward; the transition matrix at day *t* is estimated only from days ≤ *t*.
- **The persistence bar.** Every result is reported against persistence and naive baselines, not in isolation.
- **Honesty gates as code.** The engine emits loud warnings on its own output — persistence ties,
  overfit gaps, single-regime windows, low-confidence agent forecasts — and they appear in every report.
- **One engine, no special-casing.** Baseline, buy-and-hold, and the optional LLM agent are all
  pluggable `PolicyContext → position` functions through the same fill/cost path.
- **Reproducible.** Data is pinned to local snapshots and hashed; runs are deterministic and recorded.

It is explicitly **not** a money-maker, **not** self-learning, and makes **no** alpha claim.

## Safety & scope

- **Simulation only.** Paper ledger. No real funds, no orders, no broker, no wallets, no keys, no on-chain execution — ever.
- **LLM agent is CLI-only** and never publicly triggerable on your key — enforced by
  [`tests/test_agent_invariants.py`](tests/test_agent_invariants.py).
- **Keys come from the environment only** (`.env`, never committed). Binance and Asia-centric exchange APIs are out of scope.

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # all defaults work offline; no key needed for the cached path

make eval                     # the headline: daily walk-forward on BTC-USD (uses the pinned snapshot)
make robust                   # reproduce the 20-asset / +0.000 result
make test                     # 52 tests
make site                     # serve the website at http://localhost:8000
```

`make eval` reproduces the receipt: `hit_rate == persistence_hit_rate` (edge +0.000), matching the
recorded run in [`results/BTC-USD_4a150b23.json`](results/BTC-USD_4a150b23.json).

## Reproducibility

The credibility claim is *byte-reproducibility from a fresh clone*:

- **Pinned data** — `data/snapshots/` holds the exact yfinance pulls; `history_hash` stamps them.
- **Deterministic records** — every run writes a JSON record to `results/` (and `results/robustness/`).
- **Locked deps** — `requirements.lock` pins 77 packages.

Both `data/snapshots/` and `results/` are committed for this reason. `make robust` regenerates the
robustness set and the 0/5,425-divergence identity holds.

## Optional: the LLM agent (Phase C)

The agent plugs in as a *third policy*, decided by an LLM, through the identical engine — provider
switch `anthropic | ollama | none`, with a prompt-hash response cache so a whole test window costs only
a few unique model calls. Its decisions are reported with a **provenance histogram** (how each position
was parsed) and a **low-confidence gate**, never a self-asserted accuracy. The foundation is built and
offline-verified; the live agent row is pending a reachable model:

```bash
evaluate-cli --provider ollama      # free, local (Qwen on Ollama)
evaluate-cli --provider anthropic   # hosted; needs ANTHROPIC_API_KEY in .env
```

## Documentation

Current (post-reframe):

- [docs/13 — reframed plan](docs/13-reframed-plan.md) — the thesis: honest harness, null when null
- [docs/14 — Phase A requirements](docs/14-requirements-phase-a.md) — the walk-forward protocol
- [docs/15 — Phase A results](docs/15-phase-a-results.md) — the BTC held-out finding
- [docs/16 — robustness study](docs/16-robustness-study.md) — the +0.000 receipt, 20 assets
- [docs/17 — review + landscape](docs/17-review-and-landscape.md) — deep review, positioning
- [docs/12 — architecture review](docs/12-architecture-review.md) — headline-integrity fixes
- [docs/STATUS.md](docs/STATUS.md) — living status / session handoff

Legacy (pre-reframe build docs, the now-quarantined intraday path): [docs/01–11](docs/).

## Status & limitations

Phases 0–2 + Reframe A/B are done; Phase C foundation is done (live agent row pending a model). The
legacy 15s intraday loop is quarantined. Known limit: one ~3-year window per asset — a longer
multi-cycle history is the next test. Treat the **sign** of the finding (zero edge) as the result,
not the exact magnitude of the returns.

## License

[MIT](LICENSE) © 2026 Ahmad Ferdaws Shafiq. Built with [Claude Code](https://claude.com/claude-code).
