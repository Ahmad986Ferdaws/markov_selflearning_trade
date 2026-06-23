# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); the project is a portfolio /
research artifact rather than a released package, so versions are snapshots.

## [0.1.0] — 2026-06-23 — portfolio snapshot

The honest evaluation harness and its verified null result.

### The finding
- **Markov regime model has no edge over persistence.** On held-out BTC-USD (n = 328), model
  hit-rate **90.9% ≡ persistence 90.9%**. Cross-asset, the edge is **+0.000 on 20/20 assets**,
  re-verified as a pointwise identity (**0 divergences / 5,425 predictions**). Net of costs both
  trading policies lose (baseline −41.1%, buy-and-hold −47.7%).

### Added
- **Phase 0 — regime module:** Markov transition matrix, z-score regime labels, deterministic
  walk-forward (`app/services/regime.py`, `regime-cli`).
- **Reframe Phase A — daily evaluation:** causal walk-forward, train/test split, accuracy vs
  **naive and persistence** baselines, buy-and-hold yardstick, honesty gates as code
  (`app/services/evaluation.py`, `evaluate-cli`). Passed an adversarial 4-auditor review.
- **Reframe Phase B — one engine + reproducibility:** pluggable `PolicyContext → position`
  policies through a single fill/cost path; data pinned + hashed (`app/services/data_cache.py`);
  deterministic run records in `results/`.
- **Phase C — LLM agent foundation (offline):** agent as a third pluggable policy; provider switch
  `anthropic | ollama | none` (keys from env only); prompt-hash response cache
  (`app/services/agent_policy.py`).
- **Phase C — honesty hardening:** per-decision provenance histogram (`AgentStats`), a sixth gate
  warning on near-uniform forecasts, and enforcement tests (CLI-only surface, env-only keys,
  `day_index` invariance) in `tests/test_agent_invariants.py`.
- **Cross-asset robustness study:** 20 assets + 4 sub-periods (`scripts/robustness_study.py`,
  `results/robustness/`, `docs/16`).
- **Deep review + landscape positioning:** `docs/17`; SRI hashes on CDN scripts, hardened reply
  parser, `requirements.lock` (77 pins).
- **Website:** single-file animated site in the Robinhood 2024 brand language with the **real**
  held-out equity series (`design/`), plus proof media (`media/`).
- **Tests:** 52 passing.

### Quarantined
- Legacy 15-second intraday loop (`comparison.py`, `runner.py`) carries deprecation headers; the
  canonical path is the daily evaluation harness.

### Pending
- Live Phase C agent row (gated on a reachable model — Ollama or `ANTHROPIC_API_KEY`).
