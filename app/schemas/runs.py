from typing import Literal

from pydantic import BaseModel, Field


class CreateRunRequest(BaseModel):
    strategy: Literal["baseline", "agent"] = "baseline"
    ingestion_provider: Literal["yfinance", "dexscreener"] = "yfinance"
    watchlist: list[str] = Field(default_factory=lambda: ["BTC-USD"])
    trade_symbol: str | None = None


class RunStatusResponse(BaseModel):
    id: int
    strategy: str
    ingestion_provider: str
    status: str
    watchlist: list[str]
    trade_symbol: str
    cash: float
    position_qty: float
    unrealized_pnl: float
    realized_pnl: float


class TradeRecordResponse(BaseModel):
    id: int
    symbol: str
    intent_action: str
    intent_percentage: float
    enforced_action: str
    enforced_percentage: float
    quantity: float
    execution_price: float
    fees_paid: float
    guard_note: str | None
    created_at: str


class StrategyMetricsResponse(BaseModel):
    """One leg of the legacy comparison replay (mirrors comparison.StrategyMetrics)."""

    name: str
    total_return: float
    annual_sharpe: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    total_costs: float
    guard_interventions: int
    num_closes: int
    regimes_seen: int


class ComparisonResponse(BaseModel):
    """GET /runs/{id}/comparison: baseline vs agent on identical snapshots."""

    report_text: str
    headline_agent_minus_baseline: float
    baseline: StrategyMetricsResponse
    agent: StrategyMetricsResponse
    snapshot_count: int
    sharpe_periods_per_year: float
    warnings: list[str]
