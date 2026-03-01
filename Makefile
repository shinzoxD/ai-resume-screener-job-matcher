.PHONY: install run api test eval

install:
	pip install -r requirements.txt

run:
	streamlit run app.py

api:
	uvicorn backend.main:app --reload

test:
	pytest

eval:
	python scripts/evaluate.py --dataset data/eval_pairs.jsonl --output artifacts/eval_results.csv
