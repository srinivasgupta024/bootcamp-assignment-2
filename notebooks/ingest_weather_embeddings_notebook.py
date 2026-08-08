# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Weather Embeddings Ingestion
# MAGIC %md
# MAGIC # Ingest Weather Embeddings
# MAGIC
# MAGIC ETL script to read unembedded weather documents from Lakebase (weather_documents),
# MAGIC chunk long narrative text using a sliding-window strategy (size 800, overlap 100),
# MAGIC compute 384-dimensional dense vector embeddings using sentence-transformers/all-MiniLM-L6-v2,
# MAGIC and upsert the chunk vectors into Lakebase (weather_embeddings) using psycopg2 with pgvector.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %uv pip install sentence-transformers psycopg2-binary

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Imports
import json
import logging
import os
import sys
import time
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, "/Workspace/Users/vrbsrinivas8055@gmail.com/bootcamp-assignment-2")

# Note: psycopg2 and sentence_transformers imports are deferred to avoid kernel crashes
# They are imported in the cells where they're actually needed

# COMMAND ----------

# DBTITLE 1,Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest-weather-embeddings")

# Configuration parameters
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DOCS_TABLE_NAME = os.environ.get("WEATHER_DOCS_TABLE", "weather_documents")
EMBEDDINGS_TABLE_NAME = os.environ.get("WEATHER_EMBEDDINGS_TABLE", "weather_embeddings")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 100))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))

# COMMAND ----------

# DBTITLE 1,Text chunking function
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Sliding-window text chunker.
    If text length is within chunk_size, returns [text].
    Otherwise produces overlapping character chunks.
    """
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

# COMMAND ----------

# DBTITLE 1,Fetch unembedded documents
def fetch_unembedded_documents(conn, limit: int = 500) -> List[Dict]:
    """
    Fetch documents from weather_documents that do not yet have corresponding
    rows in weather_embeddings.
    """
    query = f"""
        SELECT d.id, d.location, d.source_type, d.headline, d.narrative_text
        FROM {DOCS_TABLE_NAME} d
        LEFT JOIN {EMBEDDINGS_TABLE_NAME} e ON d.id = e.document_id
        WHERE e.document_id IS NULL
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(query, (limit,))
        rows = cur.fetchall()
        return rows

# COMMAND ----------

# DBTITLE 1,Upsert embeddings batch
def upsert_embeddings_batch(conn, records: List[Tuple]):
    """
    Upsert embedding records into Lakebase using execute_values and pgvector %s::vector cast.
    Record tuple shape: (id, document_id, chunk_index, chunk_text, embedding_vector_str, model_name)
    """
    from psycopg2.extras import execute_values
    
    if not records:
        return 0

    query = f"""
        INSERT INTO {EMBEDDINGS_TABLE_NAME} (
            id, document_id, chunk_index, chunk_text, embedding, model_name, created_at
        )
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            chunk_text = EXCLUDED.chunk_text,
            embedding = EXCLUDED.embedding,
            model_name = EXCLUDED.model_name,
            created_at = now()
    """

    # Template using %s::vector for casting Python string representation of vector
    template = "(%s, %s, %s, %s, %s::vector, %s, now())"

    with conn.cursor() as cur:
        execute_values(cur, query, records, template=template, page_size=100)
        conn.commit()

    return len(records)

# COMMAND ----------

# DBTITLE 1,Main embedding pipeline
def run_embedding_pipeline():
    """Main execution entrypoint for vectorizing weather documents."""
    # Import psycopg2 inline to avoid kernel crash
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from contextlib import contextmanager
    import base64
    from databricks.sdk import WorkspaceClient
    
    # Get Lakebase connection URL
    def get_lakebase_url():
        url = os.environ.get("LAKEBASE_URL")
        if url:
            return url
        try:
            w = WorkspaceClient()
            scope = os.environ.get("LAKEBASE_SECRET_SCOPE", "database")
            key = os.environ.get("LAKEBASE_SECRET_KEY", "lakebase-url")
            secret = w.secrets.get_secret(scope=scope, key=key)
            return base64.b64decode(secret.value).decode("utf-8")
        except Exception:
            raise RuntimeError("LAKEBASE_URL environment variable or Databricks secret lookup is required.")
    
    @contextmanager
    def get_connection():
        conn = psycopg2.connect(get_lakebase_url(), cursor_factory=RealDictCursor)
        try:
            yield conn
        finally:
            conn.close()
    
    def ensure_tables(conn):
        """Create weather tables if they don't exist."""
        with conn.cursor() as cur:
            # Create weather_documents table
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
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{DOCS_TABLE_NAME}_location ON {DOCS_TABLE_NAME} (location)")
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{DOCS_TABLE_NAME}_source_type ON {DOCS_TABLE_NAME} (source_type)")
            
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create weather_embeddings table
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
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{EMBEDDINGS_TABLE_NAME}_doc_id ON {EMBEDDINGS_TABLE_NAME} (document_id)")
            
            # Create HNSW index for vector search
            try:
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{EMBEDDINGS_TABLE_NAME}_hnsw ON {EMBEDDINGS_TABLE_NAME} USING hnsw (embedding vector_cosine_ops)")
            except Exception:
                pass
            
            conn.commit()
    
    logger.info("Initializing weather database tables...")
    with get_connection() as conn:
        ensure_tables(conn)

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    logger.info("Connecting to Lakebase PostgreSQL...")
    with get_connection() as conn:
        unembedded_docs = fetch_unembedded_documents(conn, limit=1000)
        logger.info(f"Found {len(unembedded_docs)} unembedded weather documents.")

        if not unembedded_docs:
            logger.info("All documents are already embedded!")
            return

        total_chunks = 0
        records_to_insert = []

        for doc in unembedded_docs:
            doc_id = doc["id"]
            text = doc["narrative_text"]
            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

            if not chunks:
                continue

            # Compute embeddings for all chunks of this document
            embeddings = model.encode(chunks, show_progress_bar=False)

            for idx, (chunk_str, emb) in enumerate(zip(chunks, embeddings)):
                emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                # Convert list to JSON string for pgvector casting in psycopg2
                vector_str = json.dumps(emb_list)
                chunk_id = f"{doc_id}_{idx}"
                records_to_insert.append((
                    chunk_id,
                    doc_id,
                    idx,
                    chunk_str,
                    vector_str,
                    EMBEDDING_MODEL_NAME
                ))

            if len(records_to_insert) >= BATCH_SIZE:
                inserted = upsert_embeddings_batch(conn, records_to_insert)
                total_chunks += inserted
                logger.info(f"Upserted batch of {inserted} chunk embeddings (Total: {total_chunks}).")
                records_to_insert = []

        if records_to_insert:
            inserted = upsert_embeddings_batch(conn, records_to_insert)
            total_chunks += inserted
            logger.info(f"Upserted final batch of {inserted} chunk embeddings.")

        logger.info(f"Successfully finished embedding pipeline. Total chunk embeddings saved: {total_chunks}")

# COMMAND ----------

# DBTITLE 1,Run the pipeline
start_time = time.time()
run_embedding_pipeline()
logger.info(f"Pipeline finished in {time.time() - start_time:.2f} seconds.")

# COMMAND ----------

