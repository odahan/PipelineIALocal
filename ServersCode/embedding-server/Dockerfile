FROM python:3.10-slim

# Dépendances système minimales (certains wheels torch/transformers en ont besoin)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Libs Python (CPU)
RUN pip install --no-cache-dir flask sentence-transformers

WORKDIR /app
COPY server.py .

ENV PORT=80
EXPOSE 80
CMD ["python", "server.py"]
