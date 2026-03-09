FROM python:3.11-slim AS cpu
WORKDIR /app
RUN pip install --no-cache-dir redis==5.0.4 numpy==1.26.4
COPY shared /app/shared
COPY modules/inference /app/modules/inference
ENV PYTHONPATH=/app
ENV GPU_ENABLED=false
CMD ["python", "-m", "modules.inference.main"]
