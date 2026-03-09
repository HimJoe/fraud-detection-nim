FROM python:3.11-slim AS cpu
WORKDIR /app
RUN pip install --no-cache-dir redis==5.0.4 numpy==1.26.4 pandas==2.2.2
COPY shared /app/shared
COPY modules/data-prep /app/modules/data-prep
ENV PYTHONPATH=/app
ENV GPU_ENABLED=false
CMD ["python", "-m", "modules.data-prep.main"]
