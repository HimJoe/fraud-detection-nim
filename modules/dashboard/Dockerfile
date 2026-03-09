FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi==0.111.0 uvicorn[standard]==0.30.1 redis==5.0.4 httpx==0.27.0
COPY shared /app/shared
COPY modules/dashboard /app/modules/dashboard
ENV PYTHONPATH=/app
EXPOSE 8080
CMD ["uvicorn", "modules.dashboard.app:app", "--host", "0.0.0.0", "--port", "8080"]
