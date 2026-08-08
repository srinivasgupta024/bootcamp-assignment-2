"""
Lakebase (Databricks-managed Postgres) connection helper.

Connects using a single LAKEBASE_URL (a standard Postgres connection URL,
e.g. postgresql://role:password@host:5432/databricks_postgres?sslmode=require)
pointing at a native Postgres role with a static, non-expiring password.
This keeps setup to a single secret instead of five separate env vars.
"""

import base64
import os
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine

try:
    from databricks.sdk import WorkspaceClient
    _w = WorkspaceClient()
except Exception:
    _w = None


from dotenv import load_dotenv
load_dotenv()

_SCOPE = os.environ.get("LAKEBASE_SECRET_SCOPE", "database")
_KEY = os.environ.get("LAKEBASE_SECRET_KEY", "lakebase-url")


def _lakebase_url() -> str:
    """Fetch and decode the Lakebase connection URL from environment or Databricks secret scope."""
    url = os.environ.get("LAKEBASE_URL")
    if url:
        return url
    if _w is not None:
        try:
            secret = _w.secrets.get_secret(scope=_SCOPE, key=_KEY)
            return base64.b64decode(secret.value).decode("utf-8")
        except Exception:
            pass
    return "postgresql://student:npg_l3XFNci6VKUI@ep-quiet-hat-d8z5by3z.database.us-east-2.cloud.databricks.com/databricks_postgres?sslmode=require"




@contextmanager
def get_connection():
    """Yield a raw psycopg2 connection with a RealDictCursor factory."""
    conn = psycopg2.connect(_lakebase_url(), cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()


def get_engine():
    """Return a SQLAlchemy engine for Lakebase."""
    return create_engine(_lakebase_url())


def run_query(sql: str, params: tuple | dict | None = None) -> list[dict]:
    """Run a read query against Lakebase and return rows as list[dict]."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def run_write(sql: str, params: tuple | dict | None = None) -> int:
    """Run an INSERT/UPDATE/DELETE against Lakebase, return affected row count."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            return cur.rowcount


def ensure_weather_documents_table(table_name: str = "weather_documents"):
    """Create the weather_documents table in Lakebase if it doesn't exist."""
    run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id TEXT PRIMARY KEY,
            location TEXT NOT NULL,
            source_type TEXT NOT NULL,
            headline TEXT,
            narrative_text TEXT NOT NULL,
            issued_at TIMESTAMPTZ,
            payload JSONB NOT NULL,
            synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_location ON {table_name} (location)"
    )
    run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_source_type ON {table_name} (source_type)"
    )


def ensure_weather_embeddings_table(
    table_name: str = "weather_embeddings",
    docs_table_name: str = "weather_documents",
):
    """Create the weather_embeddings table and HNSW index in Lakebase if it doesn't exist."""
    # Ensure pgvector extension is active
    run_write("CREATE EXTENSION IF NOT EXISTS vector;")
    run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES {docs_table_name}(id) ON DELETE CASCADE,
            chunk_index INT NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding vector(384) NOT NULL,
            model_name TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_doc_id ON {table_name} (document_id)"
    )
    # Create HNSW index for fast vector cosine similarity search
    try:
        run_write(
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_hnsw ON {table_name} USING hnsw (embedding vector_cosine_ops)"
        )
    except Exception as e:
        # Fallback to ivfflat or proceed if HNSW creation syntax needs standard handling
        pass

