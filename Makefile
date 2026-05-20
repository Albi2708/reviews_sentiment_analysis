.PHONY: install run ui test eval

install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

run:
	uvicorn api:app --reload

ui:
	streamlit run ui.py

test:
	pytest

eval:
	python evaluate.py
