FROM python:3.10-slim

# Certificats TLS pour HTTPS (Hugging Face), nettoyage après install
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir flask sentence-transformers

WORKDIR /app
COPY server_rerank.py .

ENV RERANK_MODEL=BAAI/bge-reranker-v2-m3
ENV PORT=80

# (Optionnel) Cache HF persistant si tu montes un volume
# ENV HF_HOME=/root/.cache/huggingface

CMD ["python", "server_rerank.py"]
