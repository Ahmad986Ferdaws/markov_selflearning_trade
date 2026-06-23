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


def test_select_trade_snapshot_falls_back_to_first_snapshot():
    snapshots = [_snapshot("ETH-USD", 2000.0)]

    selected, symbol, price = _select_trade_snapshot(snapshots, "BTC-USD")

    assert selected is snapshots[0]
    assert symbol == "ETH-USD"
    assert price == 2000.0
