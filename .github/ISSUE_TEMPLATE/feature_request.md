---
name: Feature request
about: A new flag, gate, study, or doc
labels: enhancement
---

## What

<!-- The flag, gate, study, or doc you want, in one or two sentences. If it is a CLI
     flag, write the command line you would type. -->

## Why (which question does it help answer honestly?)

<!-- The question the harness cannot answer today, or answers less fairly than it
     could. Link the doc, test, or result file where the gap shows. -->

## Scope check

<!-- Tick what holds. CONTRIBUTING.md explains each line. -->

- [ ] Simulation only: no broker, wallet, key, exchange, or on-chain code
- [ ] No network in tests: FakeProvider and the pinned snapshots in data/snapshots/ are enough
- [ ] Honesty gates only append: no existing warning is weakened, silenced, or removed
- [ ] Any new number is reported against the persistence bar, not in isolation

## How to verify

<!-- The test that would pin it, or the command whose output would show it works.
     If a published number should change, say which one and why that is a finding. -->
