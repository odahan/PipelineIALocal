import os
from torch import set_num_threads, set_num_interop_threads
set_num_threads(int(os.getenv("TCH_THREADS", "6")))
set_num_interop_threads(2)


from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
app = Flask(__name__)
model = SentenceTransformer(MODEL_NAME)

@app.route("/embed", methods=["POST"])
def embed():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Field 'text' must be a non-empty string"}), 400
    vec = model.encode(text).tolist()  # 384 floats
    return jsonify({"model": MODEL_NAME, "embedding": vec})

@app.route("/ready", methods=["GET"])
def ready():
    return jsonify({"status": "ok", "model": MODEL_NAME})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
