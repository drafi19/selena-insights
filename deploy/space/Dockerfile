FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY selena_ml ./selena_ml
COPY web ./web
COPY artifacts/v0.3.0 ./artifacts/v0.3.0
ENV SELENA_ARTIFACT_DIR=/app/artifacts/v0.3.0
EXPOSE 7860
CMD ["uvicorn", "selena_ml.serve.app:app", "--host", "0.0.0.0", "--port", "7860"]
