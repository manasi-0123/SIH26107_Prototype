import re
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory

import rag

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
UI_FILE = "index.html"

print("Loading BIS RAG...")
chunks = rag.load_chunks()

if not chunks:
    raise RuntimeError("No BIS chunks found. Check the data folder.")

print(f"Total: {len(chunks)} chunks")

# Build/load the same Gemini Embedding + ChromaDB index used by rag.py.
gemini, collection = rag.build_vector_index(chunks)

print("RAG backend ready.")
print("Open http://127.0.0.1:5000 in your browser.")


@app.route("/")
def home():
    return send_from_directory(BASE_DIR, UI_FILE)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Question is required."}), 400

    try:
        history = data.get("history") or []

        # Turn follow-ups ("what about jewellers?") into a standalone search.
        search_q = rag.rewrite_query(gemini, question, history)
        hits = rag.retrieve(gemini, collection, search_q)

        answer_text = rag.ask_gemini(gemini, question, hits, history)

        # Only list the sources the answer actually cited.
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer_text)}
        sources = [
            {
                "id": c["id"],
                "title": c["title"],
                "source": c["source"],
                "section": c["section"],
                "url": c["url"],
                "last_checked": c["last_checked"],
                "similarity": round(s, 3),
            }
            for n, (s, c) in enumerate(hits, start=1)
            if n in cited
        ]

        return jsonify({"answer": answer_text.strip(), "sources": sources})

    except Exception as e:
        print(f"RAG error: {e}")
        return jsonify({
            "error": "The BIS assistant could not generate an answer.",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
