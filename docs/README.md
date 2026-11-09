# Documentation index

The docs are numbered in the order they were written, which is also the order the project
changed its mind. Read **13** first if you want the thesis, **16**, **18**, and **19** if
you want the evidence, and [STATUS.md](STATUS.md) if you want to know what is done.

## Current (post-reframe)

| Doc | What it holds |
|---|---|
| [13 — Reframed plan](13-reframed-plan.md) | The thesis: an honest daily evaluation harness that reports null when null is true. |
| [14 — Phase A requirements](14-requirements-phase-a.md) | The walk-forward protocol: causal states, train/test split, the persistence bar. |
| [15 — Phase A results](15-phase-a-results.md) | The first BTC held-out finding and the honesty gates that fired on it. |
| [16 — Robustness study](16-robustness-study.md) | 20 markets, 4 sub-periods: edge over persistence is +0.000 on 20/20, verified pointwise. |
| [17 — Review and landscape](17-review-and-landscape.md) | Multi-agent review, positioning against vectorbt / hmmlearn / agent papers, fixes applied. |
| [18 — Long-history study](18-long-history.md) | Up to 20 years per asset: still +0.000 on 8/8, 0 divergences across 16,773 predictions. |
| [19 — The anatomy of zero](19-anatomy-of-zero.md) | Why the zero is structural: positive controls, the dominance bound, switch-day decomposition, what the model *does* know. |
| [20 — Kalman pairs framework](20-kalman-pairs.md) | The `kalman/` research harness: Joseph-form filter, two-leg ledger, walk-forward vs five baselines; no edge after costs on SPY/QQQ. |
| [21 — Glossary](21-glossary.md) | The terms the current docs use, each pointing at the code or doc that defines it. |
| [12 — Architecture review](12-architecture-review.md) | The review that triggered the reframe; headline-integrity fixes. |
| [STATUS.md](STATUS.md) | Living status and session handoff. Updated at the end of each working session. |

## Legacy (pre-reframe build docs)

These describe the original 15-second intraday paper-trading loop, now quarantined behind
`ALLOW_LEGACY_AGENT_API` and kept only for the `/runs` API.

| Doc | What it holds |
|---|---|
| [01 — Scope](01-scope.md) | Boundaries: simulation only, yfinance only, no exchange APIs. |
| [02 — Stack and layout](02-stack-and-layout.md) | FastAPI, SQLAlchemy, the `app/` package layout. |
| [03 — Risk guards](03-risk-guards.md) | The hardcoded guards the agent cannot override. |
| [04 — Agent contract](04-agent-contract.md) | The strict JSON contract for LLM decisions. |
| [05 — Phase 0 regime](05-phase-0-regime.md) | The first Markov regime module. |
| [06 — Phase 1 loop](06-phase-1-loop.md) | Ingestion, paper ledger, the closed loop. |
| [07 — Phase 2 comparison](07-phase-2-comparison.md) | Baseline vs agent head-to-head on identical snapshots. |
| [08 — Build order](08-build-order.md) | Acceptance gates per phase. |
| [09 — Architecture blueprint](09-architecture-blueprint.md) | Read and write wiring. |
| [10 — Future work](10-future-work.md) | Deferred items after v1. |
| [11 — API comparison](11-api-comparison.md) | Survey of paper-trading and backtest APIs. |

A short [glossary](21-glossary.md) defines the terms the current docs use.

## Folder notes

The folders that hold evidence or tooling have their own short README:
[scripts](../scripts/README.md) (each study's inputs, outputs and network needs),
[results](../results/README.md) (the two record-naming schemes and what each directory holds),
[data](../data/README.md) (the pinned snapshots and the hash that names a run) and
[design](../design/README.md) (how the site is served, deployed and kept pinned).
