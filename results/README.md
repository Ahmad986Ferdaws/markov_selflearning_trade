# results/ — the run records

The records in this folder are the evidence the README and the docs quote. They are
committed on purpose: a fresh clone can run the same evaluation and compare against them.
They are written by the CLIs and the study scripts, never by hand.

## Two naming schemes

- **`BTC-USD_4a150b23.json`** is the headline record. Its name is the first eight characters
  of its data hash (`history_hash` over the pinned BTC-USD snapshot). It was written under
  the older, data-hash scheme, it is kept as historical evidence, and no new run can land on
  its path.
- **`{symbol}_{report_sha256}.json`** is what `evaluate-cli` writes today. The name is the
  sha256 of the complete report, so an identical report reuses its path and two different
  reports on the same price history stay separate (README, "Reproducibility").

## The study directories

| Directory | Per-asset files | `summary.json` | Pinned by |
|---|---|---|---|
| `robustness/` | one full evaluation report per asset for the 20 three-year runs, plus `BTC-USD:early` / `:late` and `SPY:early` / `:late` for the sub-periods. Written by [`scripts/robustness_study.py`](../scripts/robustness_study.py). | `aggregate` (assets, failures, how many beat persistence or buy-and-hold, mean and max edge), `assets` (20 rows), `temporal` (4 rows). `summary.csv` holds the per-asset rows as a flat table; the current script writes only the JSON, so the CSV does not change on a re-run. | [`tests/test_committed_results_records.py`](../tests/test_committed_results_records.py) |
| `long_history/` | `{"summary": row, "report": full report}` per asset for the 8 long spans. Written by [`scripts/long_history_study.py`](../scripts/long_history_study.py). | `aggregate` (8 assets, 11,348 predictions, 0 divergences, max edge, how many baselines beat hold) and the `assets` rows. | [`tests/test_committed_results_records.py`](../tests/test_committed_results_records.py) |
| `model_logic/` | none | `aggregate` over 28 runs (every visited row dominant, minimum visited diagonal, maximum hazard with support, how often the duration model fired and with what edge) and `runs`. Written by [`scripts/model_logic_study.py`](../scripts/model_logic_study.py). | [`tests/test_logic_depth_summary_claims.py`](../tests/test_logic_depth_summary_claims.py) (the duration model: 23 attempts on 5 of 28 runs) |
| `logic_depth/` | none | `aggregate` over the same 28 runs (16,773 pooled predictions, 0 divergences, the 95% rule-of-three bound, minimum prefix self-probability 0.600, log-loss and Brier improvements, memory-2 counts) and `runs`. Written by [`scripts/logic_depth_study.py`](../scripts/logic_depth_study.py). | [`tests/test_instrument_validation.py`](../tests/test_instrument_validation.py) (the never-fired uniform fallback) |

Every report carries its `data_hash`, so a record can always be matched to the snapshot it
came from.

## The rule

A changed number in this folder is a finding. It goes in the pull-request body and in
`CHANGELOG.md`, next to the data hash that produced it; it never arrives as a silent diff.
`.gitattributes` marks `results/**/*.json` and `results/**/*.csv` as generated, so GitHub
collapses them in a pull-request diff. That is the reason the words have to carry the change.
