# scripts/ — the studies

Every study runs the same honest evaluation the headline uses and writes a record under
`results/`; the docs quote the records, never a terminal. Run them from the repository root
with the project venv active (`.venv/bin/python scripts/<name>.py`); the paths inside are
relative to the root. CI runs only the re-derivation; the studies are run by hand.

| Script | Purpose | Reads | Writes | Network | Feeds | Exit codes |
|---|---|---|---|---|---|---|
| [independent_rederivation.py](independent_rederivation.py) | Re-derives the BTC-USD headline (328 held-out days, 30 switch days, 0 switch attempts, 298 hits) from the documented rules with zero `app` imports | `data/snapshots/BTC-USD_3y.pkl` (or `--snapshot PATH`), hash-checked against the headline record's data hash before scoring | nothing | never; it reads the pickle directly | [docs/19](../docs/19-anatomy-of-zero.md); run by CI and by `tests/test_independent_rederivation_script.py` | `0` reproduced, `1` mismatch, `2` snapshot changed |
| [robustness_study.py](robustness_study.py) | The 20-asset, 3-year study plus the BTC and SPY early/late halves | the 3-year snapshots through `load_or_fetch` | `results/robustness/<symbol>.json` and `summary.json` | only when a snapshot is missing | [docs/16](../docs/16-robustness-study.md) | none defined; a failed asset is counted in `n_failed` |
| [long_history_study.py](long_history_study.py) | The same protocol on the longest Yahoo span per asset (BTC 12y, ETH 10y, six assets at 20y), with the pointwise divergence count | the long snapshots through `load_or_fetch` | `results/long_history/<symbol>.json` and `summary.json` | only when a snapshot is missing | [docs/18](../docs/18-long-history.md) | none defined |
| [model_logic_study.py](model_logic_study.py) | Diagonal-dominance check, hazard curves and a causal duration-aware predictor on all 28 runs | the per-asset records in `results/robustness/` and `results/long_history/` for each run's chosen window, `k` and data hash; the matching snapshots | `results/model_logic/summary.json` | only when a snapshot is missing | [docs/19](../docs/19-anatomy-of-zero.md) | none defined |
| [logic_depth_study.py](logic_depth_study.py) | Rule-of-three bound on the pooled divergences, a second-order chain, and log-loss / Brier against sticky baselines, with a self-check against the engine's estimator | the same inputs as the model-logic study | `results/logic_depth/summary.json` | only when a snapshot is missing | [docs/19](../docs/19-anatomy-of-zero.md) | none defined |
| [_verify_divergence.py](_verify_divergence.py) | An independent re-implementation of selection and walk-forward that counts argmax-versus-persistence divergences on the 20-asset basket and the BTC early half | the 3-year snapshots through `load_or_fetch` | nothing; prints a table | only when a snapshot is missing | the 0 / 5,425 claim in [docs/16](../docs/16-robustness-study.md) | none defined |
| [record_demo.mjs](record_demo.mjs) | Renders the four stills and the scroll-through video of the site (`make demo`) | `design/index.html`, opened in headless Google Chrome over the DevTools protocol | `media/01-hero.png` to `media/04-results.png` and `media/regime-demo.mp4` | yes; the page loads its CDN scripts | `README.md` and the site's social card | none defined |

`record_demo.mjs` needs Node, Google Chrome at the path written in the file, and `ffmpeg` on
the `PATH`. No runtime is quoted for any script because neither the scripts nor the docs
record one.

## Protocol

The studies share one protocol so their numbers can be compared: the same grid (window 10,
20, 30 × `k` 0.2, 0.35, 0.5, 0.75) chosen on the first 70% of days, scored on the last 30%,
and 0.3% fee plus 0.5% slippage on every fill. Only the data span differs. Grid, split and
costs are constants near the top of each script; change one and it is a different study.

A snapshot is the data. Re-running a study after `evaluate-cli --refresh`, or after deleting
a pickle, is a new finding, not a reproduction: the data hash in the record changes, so the
record, its hash and the doc that quotes it change together, in the same pull request.
