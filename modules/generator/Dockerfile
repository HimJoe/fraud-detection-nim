FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ../../shared /app/shared
COPY . /app/modules/generator

WORKDIR /app

CMD ["python", "-m", "modules.generator.main"]
