"""Phase A/C CLI — daily walk-forward evaluation (accuracy + trading).

Usage:
  evaluate-cli                       # baseline + buy-and-hold (uses pinned snapshot)
  evaluate-cli --refresh             # re-pull fresh data and re-pin
  evaluate-cli --provider ollama     # add the LLM agent as a third policy (local)
  evaluate-cli --provider anthropic  # add the LLM agent (hosted; needs ANTHROPIC_API_KEY)

The provider can also be set once via LLM_PROVIDER in .env. The agent's responses
are cached to data/agent_cache/, so a re-run is free and reproducible.
"""

import sys

from app.config import get_settings
from app.services.data_cache import load_or_fetch, save_run_record
from app.services.evaluation import DEFAULT_POLICIES, evaluate, format_daily_report


def _parse_floats(s: str) -> tuple[float, ...]:
    return tuple(float(x) for x in s.split(",") if x.strip())


def _parse_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def _arg_value(flag: str) -> str | None:
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def _build_policies(settings):
    """Return (policies, agent_fn_or_None). Adds the LLM agent iff a provider is
    configured AND reachable — so results never contain a silently-flat agent."""
    from app.services.agent_policy import CACHE_DIR, ResponseCache, build_agent_policy, make_provider

    policies = dict(DEFAULT_POLICIES)
    provider = make_provider(settings)
    if provider is None:
        return policies, None

    print(f"[agent] provider={provider.id} model={provider.model} — health check...", flush=True)
    if not provider.health():
        print(f"[agent] provider '{provider.id}' unreachable; running WITHOUT the agent "
              f"(baseline + buy-and-hold only).")
        return policies, None

    safe_model = provider.model.replace(":", "-").replace("/", "-") or "model"
    cache = ResponseCache(CACHE_DIR / f"{provider.id}_{safe_model}.json")
    agent = build_agent_policy(provider, cache)
    policies["agent"] = agent
    print(f"[agent] enabled (cache: {len(cache._d)} prior responses)")
    return policies, agent


def main() -> None:
    s = get_settings()
    override = _arg_value("--provider")
    if override:
        s = s.model_copy(update={"llm_provider": override})

    refresh = "--refresh" in sys.argv
    history, source = load_or_fetch(s.regime_symbol, years=3, refresh=refresh)
    print(f"[data] {source}")

    policies, agent = _build_policies(s)

    report = evaluate(
        history,
        symbol=s.regime_symbol,
        train_frac=s.eval_train_frac,
        grid_windows=_parse_ints(s.eval_grid_windows),
        grid_k=_parse_floats(s.eval_grid_k),
        fee_pct=s.fee_pct,
        slippage_pct=s.slippage_pct,
        policies=policies,
    )
    print(format_daily_report(report))

    if agent is not None:
        agent.cache.save()
        print(f"\n[agent] unique model calls: {agent.cache.misses}  "
              f"cache hits: {agent.cache.hits}  errors: {agent.stats.errors}")
        # Decision provenance: how each agent position was actually obtained.
        # A measured histogram (not a self-asserted accuracy) — the project's
        # honesty idiom applied to the agent. Lots of salvage/hold == read the
        # agent row with suspicion.
        st = agent.stats
        if st.decisions:
            clean = st.strict_json / st.decisions
            print(f"[agent] decision quality ({st.decisions} decisions): "
                  f"strict-JSON {st.strict_json}  "
                  f"salvaged-labeled {st.salvaged_labeled}  "
                  f"salvaged-number {st.salvaged_single_number}  "
                  f"held(unparseable) {st.unparseable_hold}  "
                  f"held(error) {st.errors}   [{clean:.0%} clean]")
        if st.errors:
            print("[agent] WARNING: some calls failed mid-run and held the prior position; "
                  "treat the agent row with caution.")

    path = save_run_record(report)
    print(f"\n[record] saved -> {path}")


if __name__ == "__main__":
    main()
