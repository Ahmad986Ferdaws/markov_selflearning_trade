from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "sqlite:///./paper_trading.db"
    poll_interval_seconds: int = 15

    regime_symbol: str = "BTC-USD"
    regime_window: int = 20
    regime_k: float = 0.5
    regime_mode: str = "zscore"
    regime_bull_thresh: float = 0.02
    regime_bear_thresh: float = -0.02
    benchmark_symbol: str = "BTC-USD"

    # Phase A evaluation
    eval_train_frac: float = 0.7
    eval_grid_windows: str = "10,20,30"
    eval_grid_k: str = "0.2,0.35,0.5,0.75"

    starting_cash: float = 1000.0
    max_allocation_pct: float = 20.0
    liquidity_floor_usd: float = 50_000.0
    fee_pct: float = 0.3
    slippage_pct: float = 0.5

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"

    # Phase C — LLM agent policy (provider switch). Default "none" so evaluate-cli
    # is unchanged unless explicitly enabled. Keys come from the environment only.
    llm_provider: str = "none"  # "anthropic" | "ollama" | "none"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    agent_temperature: float = 0.0
    # Hard ceiling on unique (billed) LLM calls per evaluation run. Bucketing
    # normally collapses a full window to <10 unique calls; if an asset's
    # distribution defeats the bucketing, we fail CLOSED (hold) rather than fan
    # out unbounded network calls on the user's key.
    agent_max_calls: int = 64
    # Replay-only mode: serve ONLY the persisted response cache, never call a
    # live provider. This is what any public-facing surface must use.
    agent_cache_only: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
