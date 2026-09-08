from datetime import datetime, timezone

from app.schemas.snapshot import NormalizedSnapshot
from app.services.runner import _select_trade_snapshot


def _snapshot(symbol: str, price: float) -> NormalizedSnapshot:
    return NormalizedSnapshot(
        symbol=symbol,
        price=price,
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def test_select_trade_snapshot_prefers_trade_symbol():
    snapshots = [_snapshot("ETH-USD", 2000.0), _snapshot("BTC-USD", 100.0)]

    selected, symbol, price = _select_trade_snapshot(snapshots, "BTC-USD")

    assert selected is snapshots[1]
    assert symbol == "BTC-USD"
    assert price == 100.0


def test_select_trade_snapshot_never_substitutes_another_symbol():
    """Review fix: the old fallback traded whatever symbol was polled first,
    silently mixing instruments into one position. Now: no match -> no trade."""
    snapshots = [_snapshot("ETH-USD", 2000.0)]

    selected, symbol, price = _select_trade_snapshot(snapshots, "BTC-USD")

    assert selected is None
    assert symbol == "BTC-USD"
    assert price is None
