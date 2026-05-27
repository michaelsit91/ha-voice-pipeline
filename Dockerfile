FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pipeline/ ./pipeline/
CMD ["sh", "-c", "uvicorn pipeline.server:app --host 0.0.0.0 --port ${PORT:-18795}"]
