"""
Ingest Weather Embeddings (notebooks/ingest_weather_embeddings.py)

ETL script to read unembedded weather documents from Lakebase (weather_documents),
chunk long narrative text using a sliding-window strategy (size 800, overlap 100),
compute 384-dimensional dense vector embeddings using sentence-transformers/all-MiniLM-L6-v2,
and upsert the chunk vectors into Lakebase (weather_embeddings) using psycopg2 with pgvector.
"""

import json
import logging
import os
import sys
import time
from typing import List, Dict, Tuple

import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

# Add parent directory to path so lakebase helper module can be imported cleanly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import lakebase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest-weather-embeddings")

# Configuration parameters
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DOCS_TABLE_NAME = os.environ.get("WEATHER_DOCS_TABLE", "weather_documents")
EMBEDDINGS_TABLE_NAME = os.environ.get("WEATHER_EMBEDDINGS_TABLE", "weather_embeddings")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 100))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))


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


def upsert_embeddings_batch(conn, records: List[Tuple]):
    """
    Upsert embedding records into Lakebase using execute_values and pgvector %s::vector cast.
    Record tuple shape: (id, document_id, chunk_index, chunk_text, embedding_vector_str, model_name)
    """
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


def run_embedding_pipeline():
    """Main execution entrypoint for vectorizing weather documents."""
    logger.info("Initializing weather database tables...")
    lakebase.ensure_weather_documents_table(DOCS_TABLE_NAME)
    lakebase.ensure_weather_embeddings_table(EMBEDDINGS_TABLE_NAME, DOCS_TABLE_NAME)

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    logger.info("Connecting to Lakebase PostgreSQL...")
    with lakebase.get_connection() as conn:
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


if __name__ == "__main__":
    start_time = time.time()
    run_embedding_pipeline()
    logger.info(f"Pipeline finished in {time.time() - start_time:.2f} seconds.")
