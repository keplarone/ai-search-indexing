FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y curl lsof && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8741

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8741", "--workers", "4"]
