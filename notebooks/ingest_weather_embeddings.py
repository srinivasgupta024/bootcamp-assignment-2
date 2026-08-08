# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest Weather Documents -> Vector Embeddings (Lakebase)
# MAGIC
# MAGIC This notebook harvests weather alerts and forecast discussions from the National Weather Service API,
# MAGIC splits long narrative descriptions into overlapping chunks, generates vector embeddings using sentence-transformers,
# MAGIC and writes the embeddings into Lakebase (PostgreSQL with pgvector) for RAG and semantic retrieval.

# COMMAND ----------

# DBTITLE 1, Install required dependencies
# MAGIC %pip install -q 'databricks-sdk>=0.118.0' sentence-transformers requests pandas psycopg2-binary python-dotenv

# COMMAND ----------

try:
    dbutils.library.restartPython()
except NameError:
    pass

# COMMAND ----------

# DBTITLE 1, Notebook Widgets & Configuration
import json
import logging
import os
import sys
import time
import base64
from urllib.parse import urlparse
from typing import List, Dict, Tuple

import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

# Add parent directory to sys.path so helper modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest-weather-embeddings")

# Use dbutils widgets if running inside Databricks, otherwise fallback to env vars
try:
    dbutils.widgets.text("weather_docs_table", "weather_documents", "Docs Table")
    dbutils.widgets.text("weather_embeddings_table", "weather_embeddings", "Embeddings Table")
    dbutils.widgets.text("embedding_model", "sentence-transformers/all-MiniLM-L6-v2", "Embedding Model")
    dbutils.widgets.text("chunk_size", "800", "Chunk Size")
    dbutils.widgets.text("chunk_overlap", "100", "Chunk Overlap")

    DOCS_TABLE_NAME = dbutils.widgets.get("weather_docs_table")
    EMBEDDINGS_TABLE_NAME = dbutils.widgets.get("weather_embeddings_table")
    EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")
    CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))
    CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))
except Exception:
    DOCS_TABLE_NAME = os.environ.get("WEATHER_DOCS_TABLE", "weather_documents")
    EMBEDDINGS_TABLE_NAME = os.environ.get("WEATHER_EMBEDDINGS_TABLE", "weather_embeddings")
    EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 800))
    CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 100))

BATCH_SIZE = 64

# COMMAND ----------

# DBTITLE 1, Connection Helper to Lakebase
def get_lakebase_connection():
    """Resolve Lakebase connection string from Databricks Secrets or env var."""
    url = os.environ.get("LAKEBASE_URL")
    if not url:
        try:
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient()
            secret = w.secrets.get_secret(scope="database", key="lakebase-url")
            url = base64.b64decode(secret.value).decode("utf-8")
        except Exception as err:
            logger.warning(f"Databricks secret lookup failed: {err}")

    if not url or "<your-lakebase-host>" in url:
        # Fallback to hardcoded database connection if secret holds placeholder
        url = "postgresql://student:npg_l3XFNci6VKUI@ep-quiet-hat-d8z5by3z.database.us-east-2.cloud.databricks.com/databricks_postgres?sslmode=require"

    parsed = urlparse(url)
    return psycopg2.connect(
        host=parsed.hostname,
        port=parsed.port or 5432,
        dbname=parsed.path.lstrip("/"),
        user=parsed.username,
        password=parsed.password,
        sslmode="require",
        connect_timeout=10
    )

# COMMAND ----------

# DBTITLE 1, Ensure Tables Exist in Lakebase
def ensure_tables():
    """Ensure weather_documents and weather_embeddings tables exist in Lakebase."""
    with get_lakebase_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {DOCS_TABLE_NAME} (
                    id TEXT PRIMARY KEY,
                    location TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    headline TEXT,
                    narrative_text TEXT NOT NULL,
                    issued_at TIMESTAMPTZ,
                    payload JSONB NOT NULL,
                    synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {EMBEDDINGS_TABLE_NAME} (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES {DOCS_TABLE_NAME}(id) ON DELETE CASCADE,
                    chunk_index INT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding vector(384) NOT NULL,
                    model_name TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
            try:
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{EMBEDDINGS_TABLE_NAME}_hnsw ON {EMBEDDINGS_TABLE_NAME} USING hnsw (embedding vector_cosine_ops)")
            except Exception:
                pass
            conn.commit()
            logger.info(f"Database tables {DOCS_TABLE_NAME} and {EMBEDDINGS_TABLE_NAME} are ready.")

ensure_tables()

# COMMAND ----------

# DBTITLE 1, Harvest New Weather Records from NWS API if needed
try:
    from weather_client import WeatherClient
    client = WeatherClient()
    docs = client.harvest_locations(["IL", "TX", "CA", "NY", "Chicago, IL", "Austin, TX"])
    if docs:
        with get_lakebase_connection() as conn:
            with conn.cursor() as cur:
                for doc in docs:
                    cur.execute(f"""
                        INSERT INTO {DOCS_TABLE_NAME} (
                            id, location, source_type, headline, narrative_text, issued_at, payload, synced_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, now())
                        ON CONFLICT (id) DO UPDATE SET
                            location = EXCLUDED.location,
                            source_type = EXCLUDED.source_type,
                            headline = EXCLUDED.headline,
                            narrative_text = EXCLUDED.narrative_text,
                            issued_at = EXCLUDED.issued_at,
                            payload = EXCLUDED.payload,
                            synced_at = EXCLUDED.synced_at
                    """, (
                        doc["id"], doc["location"], doc["source_type"], doc.get("headline", ""),
                        doc["narrative_text"], doc.get("issued_at"), json.dumps(doc.get("payload", {}))
                    ))
                conn.commit()
        logger.info(f"Harvested and synced {len(docs)} weather documents.")
except Exception as e:
    logger.warning(f"Could not harvest new documents: {e}")

# COMMAND ----------

# DBTITLE 1, Chunk & Embed Text Pipeline
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size // 2

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks

def run_embedding_pipeline():
    logger.info(f"Loading transformer model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    with get_lakebase_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT d.id, d.location, d.source_type, d.headline, d.narrative_text
                FROM {DOCS_TABLE_NAME} d
                LEFT JOIN {EMBEDDINGS_TABLE_NAME} e ON d.id = e.document_id
                WHERE e.document_id IS NULL
                LIMIT 1000
            """)
            unembedded_docs = cur.fetchall()

        logger.info(f"Found {len(unembedded_docs)} unembedded weather documents.")
        if not unembedded_docs:
            logger.info("All documents are embedded!")
            return

        records_to_insert = []
        total_chunks = 0

        for doc in unembedded_docs:
            doc_id = doc[0]
            text = doc[4]
            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            if not chunks:
                continue

            embeddings = model.encode(chunks, show_progress_bar=False)

            for idx, (chunk_str, emb) in enumerate(zip(chunks, embeddings)):
                emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                chunk_id = f"{doc_id}_{idx}"
                records_to_insert.append((
                    chunk_id, doc_id, idx, chunk_str, json.dumps(emb_list), EMBEDDING_MODEL_NAME
                ))

            if len(records_to_insert) >= BATCH_SIZE:
                with conn.cursor() as cur:
                    execute_values(cur, f"""
                        INSERT INTO {EMBEDDINGS_TABLE_NAME} (
                            id, document_id, chunk_index, chunk_text, embedding, model_name, created_at
                        ) VALUES %s
                        ON CONFLICT (id) DO UPDATE SET
                            chunk_text = EXCLUDED.chunk_text,
                            embedding = EXCLUDED.embedding,
                            model_name = EXCLUDED.model_name,
                            created_at = now()
                    """, records_to_insert, template="(%s, %s, %s, %s, %s::vector, %s, now())", page_size=100)
                    conn.commit()
                total_chunks += len(records_to_insert)
                records_to_insert = []

        if records_to_insert:
            with conn.cursor() as cur:
                execute_values(cur, f"""
                    INSERT INTO {EMBEDDINGS_TABLE_NAME} (
                        id, document_id, chunk_index, chunk_text, embedding, model_name, created_at
                    ) VALUES %s
                    ON CONFLICT (id) DO UPDATE SET
                        chunk_text = EXCLUDED.chunk_text,
                        embedding = EXCLUDED.embedding,
                        model_name = EXCLUDED.model_name,
                        created_at = now()
                """, records_to_insert, template="(%s, %s, %s, %s, %s::vector, %s, now())", page_size=100)
                conn.commit()
            total_chunks += len(records_to_insert)

        logger.info(f"Embeddings pipeline finished successfully. Total chunk embeddings: {total_chunks}")

run_embedding_pipeline()
