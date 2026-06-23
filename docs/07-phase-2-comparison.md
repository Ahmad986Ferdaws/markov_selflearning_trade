# Phase 2 — LLM agent + head-to-head comparison

## `services/agent.py`

- Implement the strict contract from the [agent contract doc](04-agent-contract.md).
- Inputs: truncated metrics + the regime feature from [Phase 0](05-phase-0-regime.md).
- Strict JSON parsing, one repair retry, fall back to `HOLD` on failure.

## `services/comparison.py`

- Run **baseline** and **agent** on the **identical snapshot stream**.
- Replay the same `Snapshot` rows to both strategies so the comparison is fair — not two different live windows.
- Compute for each strategy:
  - Net Sharpe
  - Max drawdown
  - Total return
  - Win rate
  - Number of trades
  - Total cost paid
  - Count of guard interventions (how often the agent was clamped or rejected)

## Report

- Produce a single comparison report (JSON from the API).
- A small markdown / plot summary is a bonus.
- **Headline number:** `agent_return - baseline_return`, net of costs.

## Acceptance check (Phase 2)

One command runs both strategies on the same data and prints the comparison table.

**Be honest if the baseline wins — that is a valid and publishable result.**
