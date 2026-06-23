# REGIME — local dev convenience. Run `make help` to see targets.
PY := .venv/bin/python
PORT ?= 8000

.PHONY: help site eval regime robust test demo clean-media

help:
	@echo "REGIME — local commands"
	@echo "  make site     serve the website at http://localhost:$(PORT)  (Ctrl-C to stop)"
	@echo "  make eval     run the daily evaluation (BTC-USD) and print the report"
	@echo "  make regime   run the Phase-0 regime report"
	@echo "  make robust   run the 20-asset robustness study (writes results/robustness/)"
	@echo "  make test     run the test suite"
	@echo "  make demo     render screenshots + a scroll-through video into media/"

site:
	@echo "Serving design/ at http://localhost:$(PORT)  —  open it in a browser, Ctrl-C to stop"
	@$(PY) -m http.server $(PORT) --directory design

eval:
	@$(PY) -m app.cli.evaluate

regime:
	@$(PY) -m app.cli.regime

robust:
	@$(PY) scripts/robustness_study.py

test:
	@$(PY) -m pytest -q

demo:
	@node scripts/record_demo.mjs

clean-media:
	@rm -rf media && echo "removed media/"
