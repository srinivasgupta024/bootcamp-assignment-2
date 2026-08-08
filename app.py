"""
Vector Weather Retrieval Service  (app.py)

Flask REST API + Web Dashboard providing:
  GET  /           — Dashboard UI
  GET  /healthz    — Health check
  POST /weather/sync    — Harvest NWS alerts & forecasts → Lakebase + embed
  POST /weather/search  — Cosine-similarity pgvector search with optional RAG
  GET  /weather/documents  — Browse raw weather_documents
  GET  /weather/embeddings — Browse weather_embeddings chunks
  GET  /weather/stats      — Document & embedding counts
"""

import logging
import os
import json as _json

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify, render_template, request
from sentence_transformers import SentenceTransformer

import lakebase
from weather_client import WeatherClient
from notebooks.ingest_weather_embeddings import run_embedding_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vector-weather-app")

app = Flask(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
WEATHER_DOCS_TABLE       = os.environ.get("WEATHER_DOCS_TABLE",       "weather_documents")
WEATHER_EMBEDDINGS_TABLE = os.environ.get("WEATHER_EMBEDDINGS_TABLE", "weather_embeddings")
EMBEDDING_MODEL_NAME     = os.environ.get("EMBEDDING_MODEL",          "sentence-transformers/all-MiniLM-L6-v2")

DEFAULT_WEATHER_LOCATIONS = [
    loc.strip()
    for loc in os.environ.get("WEATHER_LOCATIONS", "IL,TX,CA,NY,FL,Chicago, IL,Austin, TX").split(",")
    if loc.strip()
]

_embedding_model = None


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_embedding_model() -> SentenceTransformer:
    """Lazy-load the SentenceTransformer (singleton per worker process)."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        cache_kw = {} if os.name == "nt" else {"cache_folder": "/tmp/.cache/huggingface"}
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, **cache_kw)
    return _embedding_model


def ensure_weather_tables() -> None:
    """Create weather tables in Lakebase if they don't already exist."""
    lakebase.ensure_weather_documents_table(WEATHER_DOCS_TABLE)
    lakebase.ensure_weather_embeddings_table(WEATHER_EMBEDDINGS_TABLE, WEATHER_DOCS_TABLE)


def _upsert_weather_docs(documents: list[dict]) -> int:
    """Upsert a batch of harvested weather documents into Lakebase."""
    if not documents:
        return 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for doc in documents:
                cur.execute(
                    f"""
                    INSERT INTO {WEATHER_DOCS_TABLE}
                        (id, location, source_type, headline, narrative_text, issued_at, payload, synced_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (id) DO UPDATE SET
                        location       = EXCLUDED.location,
                        source_type    = EXCLUDED.source_type,
                        headline       = EXCLUDED.headline,
                        narrative_text = EXCLUDED.narrative_text,
                        issued_at      = EXCLUDED.issued_at,
                        payload        = EXCLUDED.payload,
                        synced_at      = EXCLUDED.synced_at
                    """,
                    (
                        doc["id"],
                        doc["location"],
                        doc["source_type"],
                        doc.get("headline", ""),
                        doc["narrative_text"],
                        doc.get("issued_at"),
                        _json.dumps(doc.get("payload", {})),
                    ),
                )
            conn.commit()
    return len(documents)


# ── Error handler ─────────────────────────────────────────────────────────────
@app.errorhandler(Exception)
def handle_exception(err):
    """Return all unhandled errors as clean JSON."""
    logger.exception("Unhandled exception")
    code = getattr(err, "code", 500)
    return jsonify({"error": str(err)}), (code if isinstance(code, int) else 500)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "service": "Vector Weather Retrieval Service", "model": EMBEDDING_MODEL_NAME})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/weather/sync", methods=["POST"])
def sync_weather():
    """
    POST /weather/sync
    Body (optional): {"locations": ["IL", "Chicago, IL"], "limit": 50}

    Harvests NWS alerts & forecast narratives, upserts into weather_documents,
    then runs the vector embedding pipeline automatically.
    """
    ensure_weather_tables()

    body = request.get_json(silent=True) or {}
    locations = body.get("locations") or DEFAULT_WEATHER_LOCATIONS
    if isinstance(locations, str):
        locations = [l.strip() for l in locations.split(",") if l.strip()]
    limit = int(body.get("limit", 50))

    docs = WeatherClient().harvest_locations(locations, limit_per_location=limit)
    synced = _upsert_weather_docs(docs)

    try:
        run_embedding_pipeline()
    except Exception as err:
        logger.warning("Embedding pipeline warning: %s", err)

    return jsonify({"synced": synced, "locations": locations, "total_harvested": len(docs)})


@app.route("/weather/search", methods=["POST", "GET"])
def search_weather():
    """
    POST /weather/search
    Body: {"query": "flash flood risk", "top_k": 5, "source_type": "alert", "rag": true}

    Encodes the query, runs cosine-similarity search via pgvector <=> operator,
    and optionally returns a simple RAG summary.
    """
    ensure_weather_tables()

    if request.method == "POST":
        data        = request.get_json(silent=True) or {}
        query       = data.get("query")
        top_k       = data.get("top_k", 5)
        source_type = data.get("source_type")
        rag         = data.get("rag", False)
    else:
        query       = request.args.get("query")
        top_k       = request.args.get("top_k", 5)
        source_type = request.args.get("source_type")
        rag         = request.args.get("rag", "false").lower() in ("true", "1")

    if not query or not str(query).strip():
        return jsonify({"error": "'query' is required.", "results": [], "count": 0}), 400

    query = str(query).strip()
    try:
        top_k = max(1, min(int(top_k), 20))
    except (ValueError, TypeError):
        top_k = 5

    vec_str = _json.dumps(get_embedding_model().encode(query, show_progress_bar=False).tolist())

    try:
        rows = lakebase.run_query(
            f"""
            SELECT d.id, d.location, d.source_type, d.headline, d.narrative_text,
                   e.chunk_text,
                   1 - (e.embedding <=> %s::vector) AS similarity
            FROM   {WEATHER_EMBEDDINGS_TABLE} e
            JOIN   {WEATHER_DOCS_TABLE}       d ON d.id = e.document_id
            WHERE  (%s::text IS NULL OR d.source_type = %s::text)
            ORDER  BY e.embedding <=> %s::vector
            LIMIT  %s
            """,
            (vec_str, source_type, source_type, vec_str, top_k),
        )
    except Exception as err:
        logger.error("Vector search failed: %s", err)
        return jsonify({
            "query": query, "results": [], "count": 0,
            "warning": "No results found. Run POST /weather/sync to ingest data first.",
        })

    results = [
        {
            "id":             r["id"],
            "location":       r["location"],
            "source_type":    r["source_type"],
            "headline":       r["headline"],
            "narrative_text": r["narrative_text"],
            "chunk_text":     r["chunk_text"],
            "similarity":     round(float(r["similarity"] or 0), 4),
        }
        for r in rows
    ]

    payload = {"query": query, "top_k": top_k, "count": len(results), "results": results}

    if rag and results:
        top = results[0]
        payload["rag_summary"] = (
            f"Summary for query '{query}': Highest relevance report '{top['headline']}' "
            f"for location '{top['location']}' (Similarity: {top['similarity']:.2%}). "
            f"Details: {top['chunk_text'][:280]}..."
        )

    return jsonify(payload)


@app.route("/weather/documents")
def list_weather_documents():
    """GET /weather/documents?limit=50 — browse raw weather_documents."""
    ensure_weather_tables()
    limit = int(request.args.get("limit", 50))
    rows = lakebase.run_query(
        f"SELECT id, location, source_type, headline, narrative_text, issued_at, synced_at "
        f"FROM {WEATHER_DOCS_TABLE} ORDER BY synced_at DESC LIMIT %s",
        (limit,),
    )
    return jsonify(rows)


@app.route("/weather/embeddings")
def list_weather_embeddings():
    """GET /weather/embeddings?limit=50 — browse weather_embeddings chunks."""
    ensure_weather_tables()
    limit = int(request.args.get("limit", 50))
    rows = lakebase.run_query(
        f"SELECT id, document_id, chunk_index, chunk_text, model_name, created_at "
        f"FROM {WEATHER_EMBEDDINGS_TABLE} ORDER BY created_at DESC LIMIT %s",
        (limit,),
    )
    return jsonify(rows)


@app.route("/weather/stats")
def weather_stats():
    """GET /weather/stats — document & embedding counts."""
    ensure_weather_tables()
    doc_count = lakebase.run_query(f"SELECT COUNT(*) AS count FROM {WEATHER_DOCS_TABLE}")
    emb_count = lakebase.run_query(f"SELECT COUNT(*) AS count FROM {WEATHER_EMBEDDINGS_TABLE}")
    return jsonify({
        "documents":  doc_count[0]["count"] if doc_count else 0,
        "embeddings": emb_count[0]["count"] if emb_count else 0,
        "model":      EMBEDDING_MODEL_NAME,
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_RUN_PORT", 8000))
    app.run(debug=True, host=host, port=port)