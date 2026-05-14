.PHONY: install run test eval

install:
	pip install -r requirements.txt

run:
	uvicorn api:app --reload

test:
	pytest

eval:
	python evaluate.py
