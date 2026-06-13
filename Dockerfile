# Image for the whole stack (FastAPI API + Streamlit UI). docker-compose runs
# it twice: once as the API, once as the UI. Models are pre-cached at build
# time so the first request is fast.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# CPU-only torch (no CUDA) keeps the image small; then the rest of the deps and
# the spaCy model.
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt \
 && python -m spacy download en_core_web_sm

# Pre-cache the two Cardiff models into the image. Copied alone (before the rest
# of the app) so editing api/ui/db doesn't invalidate the model-download layer.
COPY pipeline.py .
RUN python -c "import pipeline; pipeline.analyze('warmup')"

COPY api.py ui.py db.py ./

EXPOSE 8000 8501

# Default command serves the API; the UI service overrides it in compose.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
