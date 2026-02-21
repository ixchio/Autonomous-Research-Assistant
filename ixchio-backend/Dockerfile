FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy all modules
COPY main.py auth.py ./
COPY clients/ clients/
COPY core/ core/
COPY pipeline/ pipeline/

ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV CHROMA_PATH=/app/chroma_db

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
