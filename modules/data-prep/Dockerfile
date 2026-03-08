# ── GPU image (for L40S nodes) ───────────────────────────────────────────────
FROM nvcr.io/nvidia/rapidsai/base:25.02-cuda12.8-py3.11

WORKDIR /app

RUN pip install --no-cache-dir redis==5.0.4

COPY ../../shared /app/shared
COPY . /app/modules/data-prep

ENV PYTHONPATH=/app
ENV GPU_ENABLED=true

CMD ["python", "-m", "modules.data-prep.main"]

# ── CPU-only build target ─────────────────────────────────────────────────────
# To build CPU-only: docker build --target cpu -t fraud-data-prep:cpu .
FROM python:3.11-slim AS cpu

WORKDIR /app
RUN pip install --no-cache-dir redis==5.0.4 numpy==1.26.4 pandas==2.2.2

COPY ../../shared /app/shared
COPY . /app/modules/data-prep

ENV PYTHONPATH=/app
ENV GPU_ENABLED=false

CMD ["python", "-m", "modules.data-prep.main"]
