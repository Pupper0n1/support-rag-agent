.PHONY: install lint typecheck test ingest eval-retrieval eval-full grade

install:
	pip install -e ".[dev]"

lint:
	ruff check src eval tests scripts
	ruff format --check src eval tests scripts

typecheck:
	mypy src eval

test:
	pytest -q

ingest:
	python scripts/ingest_kb.py data/kb/knowledge_base.jsonl --recreate

eval-retrieval:
	python eval/run_eval.py --name $(or $(NAME),retrieval) --retrieval-only

eval-full:
	python eval/run_eval.py --name $(or $(NAME),full)

grade:
	python eval/grade.py eval/reports/$(or $(NAME),retrieval).json
