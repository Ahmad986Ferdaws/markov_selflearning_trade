"""format_daily_report is the text every receipt and the CI determinism grep
read; it had no direct tests. These pin its fallbacks and structure."""

from __future__ import annotations

from app.services.evaluation import (
    AccuracyMetrics,
    DailyReport,
    PolicyResult,
    format_daily_report,
)


def _accuracy(**overrides) -> AccuracyMetrics:
    base = dict(hit_rate=0.9, balanced_accuracy=0.8, log_loss=0.3, naive_hit_rate=0.7,
                naive_balanced=0.5, naive_log_loss=0.9, persistence_hit_rate=0.9,
                persistence_balanced=0.8, n=40, n_switch=0, switch_recall=0.0, switch_attempts=0)
    base.update(overrides)
    return AccuracyMetrics(**base)


def _report(**overrides) -> DailyReport:
    base = dict(
        symbol="SYN", chosen_window=20, chosen_k=0.5, train_size=100, test_size=40,
        regime_mix={"sideways": 0.6, "bull": 0.4},
        accuracy=_accuracy(n_switch=4),
        policies=[PolicyResult("baseline", -0.1, -0.5, -0.2, 3, 0.02)],
        train_best_score=0.85, data_hash="abcdef0123456789", warnings=[],
    )
    base.update(overrides)
    return DailyReport(**base)


def _section(text: str, header: str) -> str:
    lines = text.splitlines()
    return lines[lines.index(header) + 1]


def test_empty_regime_mix_prints_the_none_fallback():
    out = format_daily_report(_report(regime_mix={}))
    assert _section(out, "--- Regime mix (test) ---") == "  (none)"


def test_regime_mix_lists_every_state_share():
    out = format_daily_report(_report())
    assert _section(out, "--- Regime mix (test) ---") == "  sideways=60.0%, bull=40.0%"


def test_missing_data_hash_leaves_no_stray_blank_line():
    out = format_daily_report(_report(data_hash=""))
    lines = out.splitlines()
    assert not any(line.startswith("Data hash") for line in lines)
    # exactly one separator between the header block and the mix block
    i = lines.index("--- Regime mix (test) ---")
    assert lines[i - 1] == "" and lines[i - 2].startswith("Train days:")
    assert "\n\n\n" not in out


def test_data_hash_is_shown_truncated_when_present():
    out = format_daily_report(_report())
    assert "Data hash: abcdef012345 (reproducible)" in out


def test_warnings_block_only_when_there_are_warnings():
    assert "WARNINGS:" not in format_daily_report(_report())
    out = format_daily_report(_report(warnings=["gate fired"]))
    assert out.rstrip().endswith("WARNINGS:\n  - gate fired")


def test_switch_recall_reads_na_when_no_switch_days():
    out = format_daily_report(_report(accuracy=_accuracy(n_switch=0)))
    assert "model recall on them: n/a" in out
