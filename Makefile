# Double-Dip Sentinel
#
# The pipeline and the pages are separate stages on purpose: `eval` is the
# slow, honest part that produces every number, and `web` only ever inlines
# what `eval` already wrote. A page can never show a figure the pipeline did
# not produce.

PY := .venv/bin/python
SEED ?= 7
ORDERS ?= 25000

.PHONY: all setup eval export web test clean serve

all: test eval web

setup:
	python3 -m venv .venv
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -r requirements.txt

test:
	$(PY) -m pytest tests/ -q

eval:
	PYTHONHASHSEED=0 $(PY) scripts/run_eval.py --orders $(ORDERS) --seed $(SEED)

export:
	PYTHONHASHSEED=0 $(PY) scripts/export_ui.py

web: export
	$(PY) scripts/build_ui.py

serve: web
	@echo "http://localhost:8000"
	@cd web && python3 -m http.server 8000

clean:
	rm -rf web/*.html reports/*.json ui/sentinel.html ui/landing.html
