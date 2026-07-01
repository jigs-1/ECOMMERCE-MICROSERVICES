FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ARG SERVICE_MODULE
ARG SERVICE_PORT=8000

ENV SERVICE_MODULE=${SERVICE_MODULE} \
    SERVICE_PORT=${SERVICE_PORT}

CMD ["sh", "-c", "uvicorn ${SERVICE_MODULE}:app --host 0.0.0.0 --port ${SERVICE_PORT}"]
