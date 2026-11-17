# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

COPY requirements.txt .

# 1. Instala PyTorch ligero (solo CPU, ~200 MB en lugar de ~2.5 GB)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url https://pytorch.org


# 2. Instala el resto de dependencias usando la caché de pip
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY app/ ./app/

CMD ["tail", "-f", "/dev/null"]
