FROM nvcr.io/nvidia/tritonserver:24.12-py3-sdk AS gpu

WORKDIR /app

RUN pip install --no-cache-dir redis==5.0.4 numpy==1.26.4

COPY ../../shared /app/shared
COPY . /app/modules/inference

ENV PYTHONPATH=/app
ENV GPU_ENABLED=true

CMD ["python", "-m", "modules.inference.main"]

# ── CPU dev target ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS cpu

WORKDIR /app
RUN pip install --no-cache-dir \
    redis==5.0.4 \
    numpy==1.26.4 \
    tritonclient[grpc]==2.44.0 || true

COPY ../../shared /app/shared
COPY . /app/modules/inference

ENV PYTHONPATH=/app
ENV GPU_ENABLED=false

CMD ["python", "-m", "modules.inference.main"]
