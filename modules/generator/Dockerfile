FROM python:3.11-slim
WORKDIR /app
COPY modules/generator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared /app/shared
COPY modules/generator /app/modules/generator
ENV PYTHONPATH=/app
CMD ["python", "-m", "modules.generator.main"]
