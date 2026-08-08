"""
ingest_weather_embeddings.py

Embedding pipeline: reads unembedded rows from weather_documents,
chunks narrative text, generates 384-dim sentence-transformer vectors,
and writes them into weather_embeddings in Lakebase (Postgres + pgvector).

Can be run standalone:
    python notebooks/ingest_weather_embeddings.py

Or imported and called from Flask:
    from notebooks.ingest_weather_embeddings import run_embedding_pipeline
    run_embedding_pipeline()
"""

import base64
import json
import os
import sys
from urllib.parse import urlparse

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

# ── Allow running from the repo root or from within notebooks/ ──────────────
try:
    _current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _current_dir = os.getcwd()

sys.path.insert(0, os.path.abspath(os.path.join(_current_dir, "..")))

# ── HuggingFace cache (Linux/Databricks only – avoid /tmp on Windows) ───────
if os.name != "nt":
    os.environ.setdefault("HF_HOME", "/tmp/.cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/.cache/huggingface")
    os.environ.setdefault("HF_HUB_CACHE", "/tmp/.cache/huggingface")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# ── Databricks widget support (no-op when running locally) ───────────────────
try:
    dbutils.library.restartPython()  # type: ignore[name-defined]  # noqa: F821
except NameError:
    pass

try:
    DOCS_TABLE_NAME = dbutils.widgets.get("weather_docs_table")              # type: ignore[name-defined]  # noqa: F821
    EMBEDDINGS_TABLE_NAME = dbutils.widgets.get("weather_embeddings_table")  # type: ignore[name-defined]  # noqa: F821
    EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")            # type: ignore[name-defined]  # noqa: F821
    CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))                      # type: ignore[name-defined]  # noqa: F821
    CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))                # type: ignore[name-defined]  # noqa: F821
except Exception:
    DOCS_TABLE_NAME       = os.environ.get("WEATHER_DOCS_TABLE",  "weather_documents")
    EMBEDDINGS_TABLE_NAME = os.environ.get("WEATHER_EMBEDDINGS_TABLE", "weather_embeddings")
    EMBEDDING_MODEL_NAME  = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    CHUNK_SIZE    = int(os.environ.get("CHUNK_SIZE",   800))
    CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 100))

_MODEL_DIMS = {
    "sentence-transformers/all-MiniLM-L6-v2":  384,
    "sentence-transformers/all-MiniLM-L12-v2": 384,
    "sentence-transformers/all-mpnet-base-v2":  768,
}
EMBEDDING_DIM = _MODEL_DIMS.get(EMBEDDING_MODEL_NAME, 384)

print(f"Using model {EMBEDDING_MODEL_NAME!r} -> {EMBEDDING_DIM}-dim vectors")

# ── Lakebase connection ───────────────────────────────────────────────────────
def _get_lakebase_url() -> str:
    """Resolve Lakebase URL: env var → Databricks secret → fallback."""
    url = os.environ.get("LAKEBASE_URL", "")
    if url and "<your-lakebase-host>" not in url:
        return url
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        import base64 as _b64
        secret = w.secrets.get_secret(scope="database", key="lakebase-url")
        return _b64.b64decode(secret.value).decode("utf-8")
    except Exception:
        pass
    # Hard-coded fallback for the bootcamp Lakebase instance
    return "postgresql://student:npg_l3XFNci6VKUI@ep-quiet-hat-d8z5by3z.database.us-east-2.cloud.databricks.com/databricks_postgres?sslmode=require"


_parsed = urlparse(_get_lakebase_url())
db_host     = _parsed.hostname
db_port     = _parsed.port or 5432
db_name     = _parsed.path.lstrip("/")
db_user     = _parsed.username
db_password = _parsed.password

print(f"Connection details: {db_host}:{db_port}/{db_name}")


def _connect() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=db_host, port=db_port, dbname=db_name,
        user=db_user, password=db_password, sslmode="require"
    )


# ── Schema setup ──────────────────────────────────────────────────────────────
def ensure_tables() -> None:
    """Create weather_documents and weather_embeddings tables if they don't exist."""
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {DOCS_TABLE_NAME} (
                id             TEXT PRIMARY KEY,
                location       TEXT        NOT NULL,
                source_type    TEXT        NOT NULL,
                headline       TEXT,
                narrative_text TEXT        NOT NULL,
                issued_at      TIMESTAMPTZ,
                payload        JSONB       NOT NULL,
                synced_at      TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {EMBEDDINGS_TABLE_NAME} (
                id          TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES {DOCS_TABLE_NAME}(id) ON DELETE CASCADE,
                chunk_index INT  NOT NULL,
                chunk_text  TEXT NOT NULL,
                embedding   vector({EMBEDDING_DIM}) NOT NULL,
                model_name  TEXT        NOT NULL,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        try:
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{EMBEDDINGS_TABLE_NAME}_hnsw "
                f"ON {EMBEDDINGS_TABLE_NAME} USING hnsw (embedding vector_cosine_ops)"
            )
        except Exception:
            pass
        conn.commit()
    print(f"[OK] Tables {DOCS_TABLE_NAME} and {EMBEDDINGS_TABLE_NAME} ready.")


# ── Text chunking ─────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping fixed-size chunks."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    step = max(chunk_size - overlap, chunk_size // 2)
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_embedding_pipeline() -> int:
    """
    Find unembedded weather_documents rows, chunk + embed them with
    all-MiniLM-L6-v2, and upsert into weather_embeddings.

    Returns the number of chunk-embeddings inserted.
    """
    ensure_tables()

    # Load rows that have no embedding yet
    with _connect() as conn:
        docs_df = pd.read_sql_query(
            f"""
            SELECT d.id, d.narrative_text
            FROM   {DOCS_TABLE_NAME} d
            LEFT JOIN {EMBEDDINGS_TABLE_NAME} e ON d.id = e.document_id
            WHERE  e.document_id IS NULL
            LIMIT  1000
            """,
            conn,
        )
    print(f"Loaded {len(docs_df)} unembedded weather documents.")

    if docs_df.empty:
        print("Nothing to embed.")
        return 0

    # Load model (uses OS default HF cache on Windows, /tmp on Linux)
    print(f"Loading embedding model {EMBEDDING_MODEL_NAME}...")
    cache_kw = {} if os.name == "nt" else {"cache_folder": "/tmp/.cache/huggingface"}
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, **cache_kw)

    # Chunk + embed
    chunk_rows = []
    for _, row in docs_df.iterrows():
        doc_id = row["id"]
        for c_idx, chunk in enumerate(chunk_text(row["narrative_text"])):
            emb = model.encode(chunk, show_progress_bar=False)
            chunk_rows.append({
                "id":          f"{doc_id}_{c_idx}",
                "document_id": doc_id,
                "chunk_index": c_idx,
                "chunk_text":  chunk,
                "embedding":   json.dumps(emb.tolist()),
                "model_name":  EMBEDDING_MODEL_NAME,
            })

    print(f"Computed {len(chunk_rows)} chunk embeddings.")

    if not chunk_rows:
        return 0

    # Upsert into Lakebase
    print(f"Inserting into {EMBEDDINGS_TABLE_NAME}...")
    with _connect() as conn:
        cur = conn.cursor()
        execute_values(
            cur,
            f"""
            INSERT INTO {EMBEDDINGS_TABLE_NAME}
                (id, document_id, chunk_index, chunk_text, embedding, model_name, created_at)
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                chunk_text  = EXCLUDED.chunk_text,
                embedding   = EXCLUDED.embedding,
                model_name  = EXCLUDED.model_name,
                created_at  = now()
            """,
            [
                (r["id"], r["document_id"], r["chunk_index"],
                 r["chunk_text"], r["embedding"], r["model_name"])
                for r in chunk_rows
            ],
            template="(%s, %s, %s, %s, %s::vector, %s, now())",
            page_size=100,
        )
        conn.commit()
    print(f"[OK] Inserted {len(chunk_rows)} embeddings into Lakebase.")
    return len(chunk_rows)


if __name__ == "__main__":
    run_embedding_pipeline()