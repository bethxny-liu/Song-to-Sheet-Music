.PHONY: setup services client test eval

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r services/requirements.txt
	cd client && npm install

services:
	MPLBACKEND=Agg PYTHONPATH=. .venv/bin/uvicorn services.app.main:app --reload --port 8000

client:
	cd client && npm run dev

test:
	.venv/bin/pip install -r requirements-dev.txt
	PYTHONPATH=. .venv/bin/pytest -m "not slow"

eval:
	PYTHONPATH=. .venv/bin/python -m evaluation.run
