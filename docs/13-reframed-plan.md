# Reframed Plan — v1.1 (post-review)∏

**Status:** Accepted (planning) · **Date:** 2026-06-07
**Category:** Portfolio / learning artifact (NOT a revenue product)
**Supersedes the framing in:** 01-scope, 06-phase-1-loop (intraday loop), 07-phase-2-comparison
**Driven by:** the multi-lens review (docs/12 + the honest-feedback workflow)

> Read the **Summary (read-first)** at the bottom if short on time.

---

## 1. The reframed thesis

Old thesis (dead): *"My LLM agent beats the market."* The review answered it — almost certainly no, net of costs, and a positive run is noise.

New thesis (alive): **"I built an honest evaluation harness that refuses to lie about whether a policy beats a dumb baseline."** The deliverable is the *rig and the honest result*, not a win. The likely finding — *agent ≈ baseline ≈ buy-and-hold, net of costs* — is a legitimate, publishable result and the actual portfolio payoff.

The project gets renamed (away from "self-learning" — nothing learns; it's a credibility liability). Suggested: `regime-eval` or `honest-agent-backtest`.

---

## 2. Locked scope

### In (v1.1)

- **Daily bars only.** One liquid asset (BTC-USD), ~2–3y yfinance history.
- **Three policies on identical data:** LLM agent, regime baseline, **buy-and-hold** (new, the honest yardstick).
- **One reproducible walk-forward**, net of costs, with the validation gate passing (>1 distinct regime).
- **One execution engine** (the ledger) used by both live and replay.
- **Switchable LLM provider:** `anthropic | ollama` via env var.
- **Reproducibility:** persist run config + per-step regime inputs + agent decisions, so a re-run gives the identical number.
- The honest writeup (markdown + one equity-curve plot).

### Out (v1.1) — explicitly

- ❌ The 15-second intraday loop (the source of the meaningless-headline bug).
- ❌ DEX Screener / memecoin mode.
- ❌ TradingView / any frontend.
- ❌ Real money, wallets, on-chain — permanently.
- ❌ Any "self-learning" / online-adaptation claim.
- ❌ Multi-asset, multi-strategy portfolios (v2).

---

## 3. Architecture

Data flow (daily):

```
yfinance daily history (BTC-USD, snapshotted into the run)
   └─► regime.regime_feature(history, as_of=bar_date)   # point-in-time, z-score thresholds
          └─► policy.decide(): agent | baseline | buy_and_hold
                 └─► ledger.apply_trade()                # SINGLE engine, costs + guards
                        └─► metrics + validation gate ──► comparison report (JSON + plot)
```

Key change from v1: there is **no separate live loop and replay simulator**. There is one daily stepper that walks the history bar-by-bar and drives every policy through the *same* `ledger.apply_trade`. "Live" (Phase D, optional) is the same stepper on a once-a-day cron.

### Device routing map (my-devices)


| Workload                                                                  | Box                    | Why                                                                                                     |
| ------------------------------------------------------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------- |
| Dev, backtests, FastAPI/SQLite, writeup                                   | **M3 MBP**             | Primary; featherweight load                                                                             |
| Local LLM serving for the free dev loop (Ollama/Qwen)                     | **Mac Mini M4**        | Already running Ollama; doesn't tie up the laptop; remote via Codex                                     |
| Postgres (only if SQLite outgrown) + optional daily cron daemon (Phase D) | **Mac Mini M4**        | Always-on tier                                                                                          |
| Official "for-the-record" agent run                                       | **Cloud (Claude API)** | Stronger reasoning/JSON for the published number                                                        |
| GPU / training / CUDA                                                     | **ROG — nothing**      | Honest call: there is no training or GPU-shaped work in this project. ROG stays idle / your gaming box. |


This is a laptop-class project. The ROG appearing with "nothing" is deliberate (my-devices rule 1: I checked whether off-loading everything from it is *right*, not just default — it is, because counting a Markov matrix and calling a frozen LLM are not GPU jobs).

---

## 4. Tech stack (chosen, with reasons)

- **Python 3.11 + FastAPI** — keep; already built, async, your default. (API is optional for a backtest artifact but useful for the demo surface.)
- **SQLite** — keep; zero-setup, behind the repo interface. Postgres on the Mini only if needed.
- **numpy/pandas** — keep; the regime math.
- **LLM: provider switch** — `anthropic` (cloud, official runs) | `ollama` (local Qwen on the Mini, free dev). Beats cloud-only (free iteration) and local-only (weaker JSON) by getting both; repair+HOLD fallback covers Qwen's JSON slips.
- **pytest** — keep; add the cross-engine and tz-boundary tests the review demanded.

---

## 5. Hard guards & risks

Guards (unchanged, still non-overridable): allocation cap 20%, mandatory fee+slippage, long/flat only, no negative cash, no shorts. Plus:

- **Reproducibility guard:** persist run config + history hash + per-step regime + agent decisions. No result is reported from a run that didn't store these.
- **Validation gate must pass** (>1 regime, finite metrics) before any number is published.
- **No live money. Ever.** Simulation-only is permanent, not a phase.

Risks:

- **R1 (biggest, and it's motivational not technical):** the honest result will likely be "no edge over buy-and-hold." That is the *expected, acceptable* outcome — frame it as the finding. The failure mode is *you* being disappointed for the wrong reason.
- **R2 — local Qwen JSON adherence:** mitigated by repair+HOLD and by running the official number on Claude.
- **R3 — yfinance is revisable:** mitigated by snapshotting the pulled history into the run.
- **R4 — overfitting thresholds:** mitigated by z-score thresholds, walk-forward, and *not* tuning to maximize return.

---

## 6. Verified-assumptions ledger


| Assumption                                                    | Status                            | Note                                                                                                          |
| ------------------------------------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| yfinance gives usable 2–3y daily BTC-USD history              | ✅                                 | Known-good; revisable, so we snapshot it per run                                                              |
| Daily Markov regime is a slow momentum filter, no strong edge | ✅                                 | Confirmed by review; this is *why* the deliverable is the honest null                                         |
| Local Qwen on Mini can serve the contract acceptably          | ✅                                 | You already run Qwen for tool-calling via OpenClaw; repair+HOLD backs it                                      |
| LLM cost for a daily backtest is negligible                   | ✅ (conclusion) / ⚠️ (exact price) | ~500 calls × small tokens. Conclusion holds for any plausible per-token price; exact $ unverified, immaterial |
| Engineering fixes (one engine, reproducibility) are tractable | ✅                                 | Scoped below; nothing load-bearing left at ⚠️                                                                 |


Nothing load-bearing is blocked at ⚠️.

---

## 7. Build order (front-loads the most risk)

**Phase A — Reframe to daily (retires the core validity risk). ~6–9 focused hrs.**

- One daily stepper over history; signal clock = trade clock.
- Fix threshold scaling: derive bull/bear from the rolling-return distribution (z-scores), not fixed ±2%/day, so the model actually changes state.
- Add buy-and-hold as the third policy.
- Demoable: one command prints agent vs baseline vs buy-and-hold, net of costs, validation gate green (>1 regime).

**Phase B — One engine + reproducibility (engineering integrity). ~4–6 hrs.**

- Comparison drives `ledger.apply_trade`; delete the duplicate fill math. One test feeds an identical decision stream to both paths and asserts identical cash/qty/PnL.
- Persist run config + history hash + per-step regime + agent decisions. Re-run → identical number.
- Fix the tz `as_of` day-boundary join; add a boundary test.
- Demoable: re-running yields the same headline; cross-engine test passes.

**Phase C — Provider switch + the artifact. ~4–6 hrs.∏**

- `agent.py` → `anthropic | ollama` via env. Dev free on local Qwen; official run on Claude.
- Replay stored decisions (no live LLM in the comparison).
- Produce the writeup + equity-curve plot. Rename the repo.
- Demoable: the published portfolio artifact.

**Phase D — (Optional) live daily daemon on the Mac Mini. ~3–4 hrs.**

- Same stepper on a once-a-day cron, Postgres on the Mini, remote-managed via Codex.
- Only if you want a "it's running live" story. Skip otherwise.

Total to a clean, defensible v1.1 (A–C): **~15–25 focused hours.**

---

## 8. Summary (read-first)

**What it is.** A reframe of the trading project from a (dead) "beat the market" bot into a **living, honest evaluation harness**: daily bars, one asset, three policies — LLM agent vs a regime baseline vs buy-and-hold — run through a single cost-aware paper ledger, reproducibly, with a gate that refuses to report a meaningless number. The product is the rig plus an honest writeup, almost certainly concluding *"the agent does not beat a dumb baseline net of costs"* — which is a credible, portfolio-grade result.

**Is it worth doing?** Yes — as a **portfolio/learning artifact**, not a money-maker. The money path is ~zero and partly illegal; that's off the table. The bankable payoff is career/credibility: a clean LLM-agent + evaluation-harness project at the exact moment employers pay a premium for that skill, made credible by the honesty and the golden-number tests.

**What it costs.** ~15–25 focused hours (Phases A–C). Run cost ≈ **$0** (local Qwen dev) to a few dollars (one official Claude run). No new infra required.

**Devices.** Laptop-class. **M3** does the work; **Mac Mini** serves the free local LLM (and an optional daily live daemon); **ROG sits this one out** — there's genuinely no GPU work, and inventing some would be theater.

**Biggest risk.** Not technical — *motivational*. The honest answer is probably "no edge," and the whole plan only works if you treat that null result as the deliverable instead of a disappointment. If you can't, this is the wrong project.

---

**Plan locked. Handoff:** build-mode, starting **Phase A — reframe to daily**.