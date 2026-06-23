# Agent contract (strict)

`schemas/` defines `AgentDecision`:

```json
{
  "action": "BUY" | "SELL" | "HOLD",
  "percentage": 0-100,
  "reasoning": "<short, data-backed>"
}
```

## Rules — `services/agent.py`

- The system prompt forces **raw JSON only** — no markdown, no backticks, no prose.
- Parse defensively:
  - strip accidental code fences,
  - validate against the Pydantic model.
- On parse failure:
  - one **repair retry**,
  - then fall back to `HOLD`.
- **Never crash the loop on bad LLM output.** Log every failure.

## Inputs

The agent is fed:
- the **truncated** metrics (six core fields — see ingestion), and
- the **regime feature** from Phase 0 (`state` + `p_next`).

It is **not** fed the raw provider dump (DEX Screener JSON or full yfinance frame)—only truncated fields.

## Output is post-clamped

Whatever the agent returns is passed through `services/guards.py` before reaching the ledger. The agent cannot override the guards.
