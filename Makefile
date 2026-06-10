.PHONY: setup backend frontend test eval

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r backend/requirements.txt
	cd frontend && npm install

backend:
	MPLBACKEND=Agg PYTHONPATH=. .venv/bin/uvicorn backend.app.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

test:
	.venv/bin/pip install -r requirements-dev.txt
	PYTHONPATH=. .venv/bin/pytest -m "not slow"

eval:
	PYTHONPATH=. .venv/bin/python -m algo.evaluation
