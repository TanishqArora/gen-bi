FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN apt-get update && apt-get install -y poppler-utils && rm -rf /var/lib/apt/lists/* \
&& pip install --no-cache-dir -r requirements.txt
COPY ingest.py ./
CMD ["python", "ingest.py"]