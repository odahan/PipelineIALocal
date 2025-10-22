import os
from flask import Flask, request, jsonify
from sentence_transformers import CrossEncoder

app = Flask(__name__)
MODEL_NAME = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# Chargement au démarrage (simple). Si tu préfères un chargement lazy,
# on peut déplacer ce code dans la route /rerank ou /ready.
cross = CrossEncoder(MODEL_NAME)

@app.route("/ready", methods=["GET"])
def ready():
    return jsonify({"status": "ok", "model": MODEL_NAME})

@app.route("/rerank", methods=["POST"])
def rerank():
    data = request.get_json(force=True)
    query = data.get("query", "")
    cands = data.get("candidates", [])
    top_n = int(data.get("top_n", 5))
    if not query or not cands:
        return jsonify({"error": "missing query or candidates"}), 400

    pairs = [(query, c) for c in cands]
    scores = cross.predict(pairs).tolist()

    ranked = sorted(zip(cands, scores), key=lambda x: x[1], reverse=True)[:top_n]
    return jsonify({
        "model": MODEL_NAME,
        "items": [{"text": t, "score": float(s)} for t, s in ranked]
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", "80"))
    # 0.0.0.0 indispensable pour exposer le port à l’hôte Docker
    app.run(host="0.0.0.0", port=port)
