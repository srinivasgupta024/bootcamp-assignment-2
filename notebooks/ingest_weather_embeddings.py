# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest Weather Documents -> Vector Embeddings (Lakebase)
# MAGIC
# MAGIC This notebook harvests weather alerts and forecast discussions from the National Weather Service API,
# MAGIC splits long narrative descriptions into overlapping chunks, generates vector embeddings using sentence-transformers,
# MAGIC and writes the embeddings into Lakebase (PostgreSQL with pgvector) for RAG and semantic retrieval.

# COMMAND ----------

# DBTITLE 1, Install required dependencies
# MAGIC %pip uninstall -y psycopg2 psycopg2-binary
# MAGIC %pip install -q 'databricks-sdk>=0.118.0' sentence-transformers trafilatura requests pandas

# COMMAND ----------

try:
    dbutils.library.restartPython()
except NameError:
    pass

# COMMAND ----------

# DBTITLE 1, Configure HuggingFace Cache & Environment Threads
import os

# Prevent PyTorch / OpenMP multi-threading SIGABRT crashes on Databricks driver node
os.environ["HF_HOME"] = "/tmp/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/tmp/.cache/huggingface"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# COMMAND ----------

# DBTITLE 1, Config Widgets
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

match EMBEDDING_MODEL_NAME:
    case "sentence-transformers/all-MiniLM-L6-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-MiniLM-L12-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-mpnet-base-v2":
        EMBEDDING_DIM = 768
    case _:
        EMBEDDING_DIM = 384

print(f"Using model {EMBEDDING_MODEL_NAME!r} -> {EMBEDDING_DIM}-dim vectors")

# COMMAND ----------

# DBTITLE 1, Parse Lakebase Connection Info
import base64
import sys
from urllib.parse import urlparse
from databricks.sdk import WorkspaceClient

# Add parent directory to sys.path so helper modules can be imported
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()

sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "..")))
sys.path.insert(0, current_dir)


def get_lakebase_url() -> str:
    url = os.environ.get("LAKEBASE_URL")
    if not url or "<your-lakebase-host>" in url:
        try:
            w = WorkspaceClient()
            secret = w.secrets.get_secret(scope="database", key="lakebase-url")
            url = base64.b64decode(secret.value).decode("utf-8")
        except Exception:
            pass
    if not url or "<your-lakebase-host>" in url:
        url = "postgresql://student:npg_l3XFNci6VKUI@ep-quiet-hat-d8z5by3z.database.us-east-2.cloud.databricks.com/databricks_postgres?sslmode=require"
    return url

lakebase_url = get_lakebase_url()
parsed = urlparse(lakebase_url)

db_host = parsed.hostname
db_port = parsed.port or 5432
db_name = parsed.path.lstrip('/')
db_user = parsed.username
db_password = parsed.password

print(f"Connection details: {db_host}:{db_port}/{db_name}")

# COMMAND ----------

# DBTITLE 1, Ensure Lakebase Postgres Tables & Index
import psycopg2

def ensure_tables():
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode='require'
    )
    try:
        cur = conn.cursor()
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
                embedding vector({EMBEDDING_DIM}) NOT NULL,
                model_name TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        try:
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{EMBEDDINGS_TABLE_NAME}_hnsw ON {EMBEDDINGS_TABLE_NAME} USING hnsw (embedding vector_cosine_ops)")
        except Exception:
            pass
        conn.commit()
        print(f"✅ Tables {DOCS_TABLE_NAME} and {EMBEDDINGS_TABLE_NAME} ready!")
    finally:
        conn.close()

ensure_tables()

# COMMAND ----------

# DBTITLE 1, Harvest Weather Records from NWS API
import json as _json
from weather_client import WeatherClient

client = WeatherClient()
docs = client.harvest_locations(["IL", "TX", "CA", "NY", "Chicago, IL", "Austin, TX"])

if docs:
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode='require'
    )
    try:
        cur = conn.cursor()
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
                doc["narrative_text"], doc.get("issued_at"), _json.dumps(doc.get("payload", {}))
            ))
        conn.commit()
        print(f"✅ Synced {len(docs)} weather documents into {DOCS_TABLE_NAME}")
    finally:
        conn.close()

# COMMAND ----------

# DBTITLE 1, Load Unembedded Documents into pandas DataFrame
import pandas as pd

conn = psycopg2.connect(
    host=db_host,
    port=db_port,
    dbname=db_name,
    user=db_user,
    password=db_password,
    sslmode='require'
)
try:
    query = f"""
        SELECT d.id, d.location, d.source_type, d.headline, d.narrative_text
        FROM {DOCS_TABLE_NAME} d
        LEFT JOIN {EMBEDDINGS_TABLE_NAME} e ON d.id = e.document_id
        WHERE e.document_id IS NULL
        LIMIT 1000
    """
    docs_df = pd.read_sql_query(query, conn)
    print(f"Loaded {len(docs_df)} unembedded weather documents.")
finally:
    conn.close()

# COMMAND ----------

# DBTITLE 1, Chunk & Compute Embeddings using SentenceTransformer
from sentence_transformers import SentenceTransformer

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text.strip() if text else ""
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

if len(docs_df) > 0:
    print(f"Loading embedding model {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")
    
    chunk_rows = []
    for idx, row in docs_df.iterrows():
        doc_id = row["id"]
        chunks = chunk_text(row["narrative_text"], CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            continue
        embeddings = model.encode(chunks, show_progress_bar=False)
        for c_idx, (chunk_str, emb) in enumerate(zip(chunks, embeddings)):
            emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
            vector_str = json.dumps(emb_list)

            chunk_rows.append({
                "id": f"{doc_id}_{c_idx}",
                "document_id": doc_id,
                "chunk_index": c_idx,
                "chunk_text": chunk_str,
                "embedding": vector_str,
                "model_name": EMBEDDING_MODEL_NAME
            })
    print(f"Computed {len(chunk_rows)} vector chunk embeddings.")
else:
    chunk_rows = []

# COMMAND ----------

# DBTITLE 1, Write Vector Embeddings to Lakebase using psycopg2
from psycopg2.extras import execute_values

if chunk_rows:
    print(f"Inserting {len(chunk_rows)} embeddings into {EMBEDDINGS_TABLE_NAME}...")
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode='require'
    )
    try:
        cursor = conn.cursor()
        insert_data = [
            (
                row['id'],
                row['document_id'],
                row['chunk_index'],
                row['chunk_text'],
                row['embedding'],
                row['model_name']
            )
            for row in chunk_rows
        ]
        insert_sql = f"""
            INSERT INTO {EMBEDDINGS_TABLE_NAME} (
                id, document_id, chunk_index, chunk_text, embedding, model_name, created_at
            ) VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                chunk_text = EXCLUDED.chunk_text,
                embedding = EXCLUDED.embedding,
                model_name = EXCLUDED.model_name,
                created_at = now()
        """
        template = "(%s, %s, %s, %s, %s::vector, %s, now())"
        execute_values(cursor, insert_sql, insert_data, template=template, page_size=100)
        conn.commit()
        print(f"✅ Successfully inserted {len(insert_data)} chunk vector embeddings into Lakebase!")
    finally:
        cursor.close()
        conn.close()
else:
    print("No new chunk embeddings to insert.")
