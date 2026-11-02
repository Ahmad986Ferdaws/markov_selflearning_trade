## What

<!-- One or two sentences. What changes, in plain words. -->

## Why

<!-- The bug, finding, or gap this closes. Link the doc or test that motivated it. -->

## Checklist

- [ ] `make test` passes locally (the suite is fully offline: FakeProvider, pinned snapshots)
- [ ] No network call was added to a test; no new data was pulled without re-pinning `data/snapshots/`
- [ ] The headline numbers are untouched, or the change to `results/` is explained above
- [ ] Honesty gates only *append* caveats; none was weakened, silenced, or removed
- [ ] Simulation only: no broker, wallet, key, or on-chain code; credentials stay in `.env`
- [ ] `CHANGELOG.md` / docs updated if behaviour changed
