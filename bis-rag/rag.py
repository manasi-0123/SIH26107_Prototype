"""
BIS RAG prototype — Gemini Embeddings + ChromaDB.

Folder layout:
    bis-rag/
        rag.py
        data/           <- your .md files
        chroma_db/      <- created automatically

Run:
    python rag.py
    python rag.py "What is HUID?"

Install once:
    python -m pip install -U google-genai chromadb

The same GEMINI_API_KEY is used for both Gemini answers and Gemini embeddings.
"""

import hashlib
import os
import re
import sys
from pathlib import Path

import chromadb
from google import genai
from google.genai import types

MODEL = "gemini-3.5-flash-lite"  # your existing answer model
EMBED_MODEL = "gemini-embedding-2"
EMBED_DIM = 768
DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "bis_docs_v1"
TOP_K = 4
SHOW_RETRIEVED = True

sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------- loading ---

def parse_file(path):
    """Read one .md file and turn its headings into retrieval chunks.

    Supports both:
      FAQ files:  ## Q1: What is ...?
      QCO files:  ## 1. Overview of QCO
    """
    lines = path.read_text(encoding="utf-8").splitlines()

    # 1) Metadata block between the two '---' lines.
    meta = {}
    start = 0
    if lines and lines[0].strip() == "---":
        for j in range(1, len(lines)):
            if lines[j].strip() == "---":
                start = j + 1
                break
            if ":" in lines[j]:
                key, value = lines[j].split(":", 1)
                meta[key.strip().lower()] = value.strip()

    if meta.get("status", "").lower() != "current":
        print(f"  (skipping {path.name}: status is not 'current')")
        return []

    chunks = []
    cur = None

    for line in lines[start:]:
        # FAQ heading, e.g. ## Q8: What is HUID?
        m = re.match(r"^##\s+(Q\d+):\s*(.*)$", line)

        # QCO numbered heading, e.g. ## 1. Overview of QCO
        if not m:
            qco = re.match(r"^##\s+(\d+(?:\.\d+)?)\.\s+(.+)$", line)
            if qco:
                m = qco

        # QCO introductory heading, if present.
        if not m and line.strip().lower() == "## about this guidance document":
            m = re.match(r"^##\s+(.+)$", line)

        if m:
            if cur:
                chunks.append(cur)

            if m.group(1).startswith("Q"):
                chunk_id = m.group(1).strip()
                title = m.group(2).strip()
            elif re.fullmatch(r"\d+(?:\.\d+)?", m.group(1)):
                chunk_id = m.group(1).strip()
                title = m.group(2).strip()
            else:
                chunk_id = "intro"
                title = m.group(1).strip()

            cur = {
                "id": chunk_id,
                "title": title,
                "body_lines": [],
                "notes": [],
                "source": meta.get("source", path.name),
                "section": meta.get("section", ""),
                "url": meta.get("url", ""),
                "last_checked": meta.get("last_checked", ""),
            }

        elif cur is not None:
            if line.strip().lower().startswith("note:"):
                cur["notes"].append(line.strip()[5:].strip())
            else:
                cur["body_lines"].append(line)

    if cur:
        chunks.append(cur)

    for c in chunks:
        c["body"] = "\n".join(c.pop("body_lines")).strip()

    return [c for c in chunks if c["body"]]


def load_chunks():
    if not DATA_DIR.exists():
        sys.exit(f"Can't find the data folder: {DATA_DIR}")

    chunks = []
    for path in sorted(DATA_DIR.glob("*.md")):
        found = parse_file(path)
        print(f"  {path.name}: {len(found)} chunks")
        chunks.extend(found)

    return chunks


# -------------------------------------------------------------- embeddings ---

def chunk_key(chunk):
    """Stable ID so the same chunk can be updated instead of duplicated."""
    raw = f"{chunk['source']}|{chunk['section']}|{chunk['id']}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def chunk_hash(chunk):
    raw = f"{chunk['title']}\n{chunk['body']}\n{' | '.join(chunk['notes'])}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def embedding_text(chunk):
    # Include the question/section title because it is highly useful for retrieval.
    return f"{chunk['title']}\n{chunk['body']}"


def embed_texts(client, texts):
    """Embed each text separately using Gemini Embedding 2."""
    vectors = []
    batch_size = 20

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        # Gemini Embedding 2 aggregates plain strings passed together.
        # Wrapping each text as its own Content object makes each input
        # produce its own embedding vector.
        contents = [
            types.Content(parts=[types.Part.from_text(text=text)])
            for text in batch
        ]

        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=contents,
            config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
        )

        vectors.extend([embedding.values for embedding in result.embeddings])

    return vectors


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def build_vector_index(chunks):
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY is not set. See the setup steps.")

    gemini = genai.Client()
    collection = get_collection()

    # See what is already indexed so normal restarts do not re-embed everything.
    existing = collection.get(include=["metadatas"])
    existing_meta = {}
    for i, metadata in enumerate(existing.get("metadatas") or []):
        if metadata:
            existing_meta[existing["ids"][i]] = metadata

    current_ids = set()
    to_embed = []
    to_upsert = []

    for chunk in chunks:
        cid = chunk_key(chunk)
        current_ids.add(cid)
        h = chunk_hash(chunk)

        metadata = {
            "title": chunk["title"],
            "source": chunk["source"],
            "section": chunk["section"],
            "url": chunk["url"],
            "last_checked": chunk["last_checked"],
            "chunk_id": chunk["id"],
            "content_hash": h,
            # Chroma metadata values must be scalar, so notes become one string.
            "notes": " | ".join(chunk["notes"]),
        }

        old = existing_meta.get(cid)
        if old and old.get("content_hash") == h:
            continue

        to_embed.append(chunk)
        to_upsert.append((cid, metadata, embedding_text(chunk)))

    # Remove chunks that no longer exist in the Markdown data.
    stale_ids = [cid for cid in existing_meta if cid not in current_ids]
    if stale_ids:
        collection.delete(ids=stale_ids)

    if to_embed:
        print(f"Embedding {len(to_embed)} new/changed chunks with {EMBED_MODEL}...")
        vectors = embed_texts(gemini, [embedding_text(c) for c in to_embed])

        collection.upsert(
            ids=[item[0] for item in to_upsert],
            embeddings=vectors,
            documents=[item[2] for item in to_upsert],
            metadatas=[item[1] for item in to_upsert],
        )
    else:
        print("Vector index is already up to date.")

    print(f"Vector DB: {collection.count()} chunks")
    return gemini, collection


def retrieve(gemini, collection, question, k=TOP_K):
    result = gemini.models.embed_content(
        model=EMBED_MODEL,
        contents=question,
        config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
    )
    query_vector = result.embeddings[0].values

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(documents, metadatas, distances):
        chunk = {
            "id": metadata.get("chunk_id", ""),
            "title": metadata.get("title", ""),
            "body": document.split("\n", 1)[1] if "\n" in document else document,
            "notes": [metadata["notes"]] if metadata.get("notes") else [],
            "source": metadata.get("source", ""),
            "section": metadata.get("section", ""),
            "url": metadata.get("url", ""),
            "last_checked": metadata.get("last_checked", ""),
        }
        # Chroma's cosine distance is lower for more similar vectors.
        similarity = 1.0 - float(distance)
        hits.append((similarity, chunk))

    return hits


# ------------------------------------------------------------- generation ---

SYSTEM_PROMPT = """You are Insight AI, a friendly, knowledgeable assistant for the Bureau of Indian Standards (BIS). Talk like a helpful person, not a search engine.

- Greetings, thanks and small talk: reply naturally, no citations needed for these conversations.
- Do not say Hello/Hey for every text block, just at the start of the conversation
- For BIS facts, rely only on the numbered context blocks and cite them like [1]. Explain in your own words instead of copying the text.
- Use the conversation so far to understand follow-ups like "what about jewellers?" or "and the fee?".
- "Hallmark", "hallmarking", "hallmarked" etc. mean the same thing; don't treat word variants as different topics.
- If the context only partly answers, share what it does support, say what's missing, and suggest a follow-up or ask one clarifying question.
- Only if nothing relevant is in the context, say you couldn't find it in the BIS sources and mention what you can help with.
- If a block has a NOTE, pass it on as a caution.
- Be warm and clear. Short paragraphs, and bullets only when they help."""


def build_context(hits):
    blocks = []
    for n, (_, c) in enumerate(hits, start=1):
        head = f"[{n}] {c['source']} | {c['section']} | {c['id']} | last checked {c['last_checked']}"
        parts = [head, f"Question/Section: {c['title']}", c["body"]]
        parts += [f"NOTE: {note}" for note in c["notes"]]
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


def rewrite_query(gemini, question, history):
    if not history:
        return question
    convo = "\n".join(f"{t['role']}: {t['text']}" for t in history[-6:])
    prompt = (
        "Rewrite the user's last message as one standalone search query about BIS "
        "standards/certification, using the conversation to resolve pronouns and "
        "follow-ups. Output only the query.\n\n"
        f"{convo}\n\nLast message: {question}"
    )
    r = gemini.models.generate_content(model=MODEL, contents=prompt)
    return (r.text or question).strip()


def ask_gemini(gemini, question, hits, history=None):
    contents = []
    for t in (history or [])[-6:]:
        role = "user" if t["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=t["text"])]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(
        text=f"CONTEXT:\n{build_context(hits)}\n\nQUESTION: {question}")]))
    response = gemini.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    return response.text


def answer(gemini, collection, question):
    hits = retrieve(gemini, collection, question)

    if SHOW_RETRIEVED:
        print("\n  Retrieved:")
        for score, c in hits:
            print(f"   - {c['id']} ({c['section']}) similarity {score:.3f}: {c['title'][:70]}")

    if not hits:
        print("\nI couldn't find this in my BIS sources.")
        return

    try:
        text = ask_gemini(gemini, question, hits)
    except Exception as e:
        print(f"\nGemini call failed: {e}")
        return

    print("\n" + text.strip())

    # Sources are printed by the code, not by the AI, so they can't be invented.
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(hits)})
    if cited:
        print("\nSources:")
        for n in cited:
            c = hits[n - 1][1]
            print(f" [{n}] {c['source']}, {c['section']}, {c['id']}  (last checked {c['last_checked']})")
            if c["url"]:
                print(f"     {c['url']}")


def main():
    print("Loading data...")
    chunks = load_chunks()
    if not chunks:
        sys.exit("No chunks loaded. Check the data folder.")
    print(f"Total: {len(chunks)} chunks")

    try:
        gemini, collection = build_vector_index(chunks)
    except Exception as e:
        print(f"\nEmbedding/vector DB setup failed: {e}")
        print("Check that google-genai and chromadb are installed and GEMINI_API_KEY is set.")
        return

    if len(sys.argv) > 1:
        answer(gemini, collection, " ".join(sys.argv[1:]))
        return

    print("\nAsk a question (or type 'quit').")
    while True:
        q = input("\n> ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            answer(gemini, collection, q)


if __name__ == "__main__":
    main()
