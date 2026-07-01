FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV CAMEO_WARM_START=1
ENV CAMEO_REQUIRE_WEIGHTS=1
ENV CAMEO_LOCAL_FILES_ONLY=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY cameo ./cameo
COPY artifacts ./artifacts
COPY ui/dist ./ui/dist
COPY pyproject.toml README.md ./

EXPOSE 8000

CMD ["uvicorn", "cameo.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
