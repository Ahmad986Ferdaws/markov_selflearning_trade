# data/ — the pinned snapshots

## `snapshots/`

`snapshots/{symbol}_{years}y.pkl` is a yfinance pull saved by `load_or_fetch`
(`app/services/data_cache.py`): one pandas DataFrame per symbol and span, written the first
time that pair is requested and read on every run after that. The committed set is the 20
three-year series of the robustness study and the 8 long series (12, 10 and 20 years) of the
long-history study; the Kalman pairs harness reads the 20-year SPY and QQQ pins through
`kalman/data.py`.

The test suite reads only these files, never the network. `tests/test_data_cache_pinning.py`
monkeypatches `fetch_daily_history` to raise, so a run that reached for yfinance would fail.

## The hash that names a run

`history_hash` is the sha256 of the close series: each date paired with its close rounded to
eight decimals, joined into one string. It is stamped into every run record as `data_hash`,
and the headline record `results/BTC-USD_4a150b23.json` is named by its first eight
characters. Two runs with the same hash saw the same data; a different hash is different
data, whatever the symbol says.

## Refreshing a pin

`evaluate-cli --refresh` re-pulls the configured symbol and overwrites its pickle; deleting a
pickle has the same effect on the next run. Either way the result is a new data hash, so a
new record under `results/`, and every doc that quotes the old numbers has to change in the
same pull request. A refresh is a new finding, not a reproduction.

## `agent_cache/`

`data/agent_cache/` is git-ignored. It holds the LLM agent's responses keyed by a hash of the
prompt, so a re-run replays them; `--cache-only` serves nothing else and makes no live call.
Delete it and the next live run rebuilds it. No published number depends on it.

## Trust

A pickle runs code when it is loaded. Only commit pickles produced by this repository's own
code (`load_or_fetch`), never one that arrived from elsewhere, and read a pull request that
changes a `.pkl` as a data change: new hash, new record, new numbers.

The glossary defines [pinned snapshot](../docs/21-glossary.md) and
[run record](../docs/21-glossary.md) in the same terms.
