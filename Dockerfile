FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY chatbot.py .
COPY extractor.py .
COPY pipeline.py .
COPY obligaciones_iga.json .

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --timeout-keep-alive 300 --h11-max-incomplete-event-size 524288000"]