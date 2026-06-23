"""Phase 0 CLI entry point."""

from app.config import get_settings
from app.services.regime import run_phase0_report


def main() -> None:
    s = get_settings()
    print(
        run_phase0_report(
            symbol=s.regime_symbol,
            window=s.regime_window,
            bull_thresh=s.regime_bull_thresh,
            bear_thresh=s.regime_bear_thresh,
            fee_pct=s.fee_pct,
            slippage_pct=s.slippage_pct,
        )
    )


if __name__ == "__main__":
    main()
