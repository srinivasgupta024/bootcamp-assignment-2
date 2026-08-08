"""
Databricks App boilerplate:
- Serves a small Flask API
- Reads/writes to Lakebase (Databricks-managed Postgres) via lakebase.py
- Pulls data from the Massive API via massive_client.py and syncs it into Lakebase

Run locally:
    python app.py
Deploy as a Databricks App using app.yaml.
"""

import logging
import os
import re

import json as _json

import requests
from flask import Flask, jsonify, render_template, request
from sentence_transformers import SentenceTransformer

import lakebase
from massive_client import MassiveClient
from weather_client import WeatherClient
from notebooks.ingest_weather_embeddings import run_embedding_pipeline

try:
    from databricks.sdk import WorkspaceClient
    _w = WorkspaceClient()
except Exception:
    _w = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("massive-app")

app = Flask(__name__)




TABLE_NAME = os.environ.get("MASSIVE_TABLE_NAME", "massive_records")
WATCHLIST_TABLE_NAME = os.environ.get("WATCHLIST_TABLE_NAME", "watchlist")
NEWS_TABLE_NAME = os.environ.get("NEWS_TABLE_NAME", "ticker_news_documents")
WEATHER_DOCS_TABLE = os.environ.get("WEATHER_DOCS_TABLE", "weather_documents")
WEATHER_EMBEDDINGS_TABLE = os.environ.get("WEATHER_EMBEDDINGS_TABLE", "weather_embeddings")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

DEFAULT_WEATHER_LOCATIONS = [
    loc.strip() for loc in os.environ.get("WEATHER_LOCATIONS", "IL,TX,CA,NY,FL,Chicago, IL,Austin, TX").split(",") if loc.strip()
]

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model

# Tickers to fetch news for by default (comma-separated), e.g. "AAPL,MSFT,GOOGL"
DEFAULT_NEWS_TICKERS = [
    t.strip().upper()
    for t in os.environ.get("NEWS_TICKERS", "AAPL,MSFT,GOOGL,AMZN,TSLA").split(",")
    if t.strip()
]

# Basic stock ticker shape check: 1-10 uppercase letters, with an optional
# ".X" or ".XX" share-class suffix (e.g. "BRK.B"). This rejects obviously
# malformed input before we even call the Massive API.
_TICKER_RE = re.compile(r"^[A-Z]{1,10}(\.[A-Z]{1,2})?$")



def ensure_table():
    """Create the destination table in Lakebase if it doesn't exist yet."""
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id TEXT PRIMARY KEY,
            payload JSONB NOT NULL,
            synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def ensure_watchlist_table():
    """Create the watchlist table in Lakebase if it doesn't exist yet."""
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {WATCHLIST_TABLE_NAME} (
            symbol TEXT NOT NULL,
            email TEXT NOT NULL,
            latest_price NUMERIC,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (symbol, email)
        )
        """
    )


def ensure_news_table():
    """
    Create the raw ticker-news documents table in Lakebase if it doesn't
    exist yet. This is the RAW document store the Spark notebook
    (notebooks/ingest_ticker_news_embeddings.py) reads from to compute
    vector embeddings into a separate `<NEWS_TABLE_NAME>_embeddings` table.
    """
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {NEWS_TABLE_NAME} (
            id TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            author TEXT,
            article_url TEXT,
            publisher_name TEXT,
            keywords JSONB,
            sentiment TEXT,
            sentiment_reasoning TEXT,
            published_utc TIMESTAMPTZ,
            payload JSONB NOT NULL,
            synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    lakebase.run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{NEWS_TABLE_NAME}_ticker "
        f"ON {NEWS_TABLE_NAME} (ticker)"
    )


def _current_user_email() -> str:
    """
    Resolve the current user's email so the watchlist can be personalized.

    Databricks Apps inject the logged-in user's identity via the
    X-Forwarded-Email header on every request. Fall back to the Databricks
    SDK's current_user API for local development where that header isn't set.
    """
    header_email = request.headers.get("X-Forwarded-Email")
    if header_email:
        return header_email
    if _w is not None:
        try:
            return _w.current_user.me().user_name
        except Exception:
            pass
    return "local-user@example.com"



@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.errorhandler(Exception)
def handle_exception(err):
    """Ensure all unhandled errors return JSON (not an HTML error page),
    so the frontend's resp.json() call never chokes on HTML."""
    logger.exception("Unhandled exception while processing request")
    status_code = getattr(err, "code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    return jsonify({"error": str(err)}), status_code


@app.route("/")
def index():
    """Simple UI to submit a list of stock symbols to sync from Massive."""
    return render_template("index.html")


@app.route("/records")
def list_records():
    """Read records already synced into Lakebase."""
    limit = int(request.args.get("limit", 100))
    rows = lakebase.run_query(
        f"SELECT id, payload, synced_at FROM {TABLE_NAME} ORDER BY synced_at DESC LIMIT %s",
        (limit,),
    )
    return jsonify(rows)


@app.route("/sync", methods=["POST"])
def sync_from_massive():
    """
    Pull data from the Massive API (paginated, potentially huge dataset) and
    upsert it into Lakebase in batches.
    """
    ensure_table()
    client = MassiveClient()

    path = request.json.get("path", "/records") if request.is_json else "/records"
    batch_size = int(request.args.get("batch_size", 500))

    batch = []
    total = 0
    for item in client.paginated_get(path):
        batch.append(item)
        if len(batch) >= batch_size:
            total += _upsert_batch(batch)
            batch = []

    if batch:
        total += _upsert_batch(batch)

    return jsonify({"synced": total})


@app.route("/news/sync", methods=["POST"])
def sync_news_from_massive():
    """
    Pull recent news articles for a set of tickers from Massive (ONE API
    call per ticker, via MassiveClient.get_news) and upsert them into the
    ticker_news_documents table in Lakebase.

    Body (optional JSON): {"tickers": ["AAPL", "MSFT"], "limit": 50}
    Defaults to DEFAULT_NEWS_TICKERS when no tickers are supplied.
    """
    ensure_news_table()
    client = MassiveClient()

    body = request.json if request.is_json else {}
    tickers = body.get("tickers") or DEFAULT_NEWS_TICKERS
    tickers = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]
    limit = int(body.get("limit", 50))

    total = 0
    for ticker in tickers:
        if not _TICKER_RE.match(ticker):
            continue
        articles = client.get_news(ticker, limit=limit)
        total += _upsert_news_batch(ticker, articles)

    return jsonify({"synced": total, "tickers": tickers})


def ensure_weather_tables():
    """Ensure raw weather documents and weather vector embedding tables exist in Lakebase."""
    lakebase.ensure_weather_documents_table(WEATHER_DOCS_TABLE)
    lakebase.ensure_weather_embeddings_table(WEATHER_EMBEDDINGS_TABLE, WEATHER_DOCS_TABLE)


def _upsert_weather_docs_batch(documents: list[dict]) -> int:
    """Upsert weather document records into Lakebase in batch."""
    if not documents:
        return 0
    count = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for doc in documents:
                cur.execute(
                    f"""
                    INSERT INTO {WEATHER_DOCS_TABLE} (
                        id, location, source_type, headline, narrative_text, issued_at, payload, synced_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (id) DO UPDATE SET
                        location = EXCLUDED.location,
                        source_type = EXCLUDED.source_type,
                        headline = EXCLUDED.headline,
                        narrative_text = EXCLUDED.narrative_text,
                        issued_at = EXCLUDED.issued_at,
                        payload = EXCLUDED.payload,
                        synced_at = EXCLUDED.synced_at
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
                count += 1
            conn.commit()
    return count


@app.route("/weather/sync", methods=["POST"])
def sync_weather():
    """
    Harvest unstructured weather text (alerts & forecasts) from NWS API
    for specified locations, upsert records into Lakebase, and auto-run vector embedding pipeline.

    Body (optional JSON): {"locations": ["IL", "Chicago, IL", "Austin, TX"], "limit": 50}
    """
    ensure_weather_tables()
    client = WeatherClient()

    body = request.json if request.is_json else {}
    locations = body.get("locations") or DEFAULT_WEATHER_LOCATIONS
    if isinstance(locations, str):
        locations = [l.strip() for l in locations.split(",") if l.strip()]

    limit = int(body.get("limit", 50))

    docs = client.harvest_locations(locations, limit_per_location=limit)
    synced_count = _upsert_weather_docs_batch(docs)

    # Automatically compute embeddings for newly ingested documents
    embedded_chunks = 0
    try:
        run_embedding_pipeline()
    except Exception as err:
        logger.warning(f"Embedding pipeline trigger encountered error: {err}")

    return jsonify({
        "synced": synced_count,
        "locations": locations,
        "total_harvested": len(docs)
    })


@app.route("/weather/search", methods=["POST", "GET"])
def search_weather():
    """
    Perform cosine-similarity vector search against weather_embeddings using pgvector <=> operator.

    Request parameters (JSON body for POST or query params for GET):
        - query (str, required): Natural language search query
        - top_k (int, optional): Max results to return (default 5, max 20)
        - source_type (str, optional): Filter by "alert" or "forecast"
        - rag (bool, optional): Include LLM/basic RAG summary
    """
    ensure_weather_tables()

    if request.method == "POST":
        data = request.json if request.is_json else {}
        query = data.get("query")
        top_k = data.get("top_k", 5)
        source_type = data.get("source_type")
        rag = data.get("rag", False)
    else:
        query = request.args.get("query")
        top_k = request.args.get("top_k", 5)
        source_type = request.args.get("source_type")
        rag = request.args.get("rag", "false").lower() in ("true", "1")

    if not query or not str(query).strip():
        return jsonify({"error": "Query string parameter 'query' is required.", "results": [], "count": 0}), 400

    query = str(query).strip()

    try:
        top_k = int(top_k)
    except (ValueError, TypeError):
        top_k = 5
    top_k = max(1, min(top_k, 20))

    # Generate vector embedding for input search query
    model = get_embedding_model()
    query_vector = model.encode(query, show_progress_bar=False).tolist()
    query_vector_str = _json.dumps(query_vector)

    sql = f"""
        SELECT 
            d.id,
            d.location,
            d.source_type,
            d.headline,
            d.narrative_text,
            e.chunk_text,
            1 - (e.embedding <=> %s::vector) AS similarity
        FROM {WEATHER_EMBEDDINGS_TABLE} e
        JOIN {WEATHER_DOCS_TABLE} d ON d.id = e.document_id
        WHERE (%s::text IS NULL OR d.source_type = %s::text)
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s;
    """

    try:
        rows = lakebase.run_query(
            sql, (query_vector_str, source_type, source_type, query_vector_str, top_k)
        )
    except Exception as err:
        logger.error(f"Vector search query failed: {err}")
        return jsonify({
            "query": query,
            "results": [],
            "count": 0,
            "warning": "No vector search results found or weather_embeddings table is empty. Try POST /weather/sync first."
        })

    results = []
    for r in rows:
        results.append({
            "id": r.get("id"),
            "location": r.get("location"),
            "source_type": r.get("source_type"),
            "headline": r.get("headline"),
            "narrative_text": r.get("narrative_text"),
            "chunk_text": r.get("chunk_text"),
            "similarity": round(float(r.get("similarity", 0.0)), 4) if r.get("similarity") is not None else 0.0
        })

    payload = {
        "query": query,
        "top_k": top_k,
        "count": len(results),
        "results": results
    }

    if rag and results:
        top_match = results[0]
        payload["rag_summary"] = (
            f"Summary for '{query}': High relevance report ({top_match['headline']}) found for location '{top_match['location']}'. "
            f"Details: {top_match['chunk_text'][:250]}..."
        )

    return jsonify(payload)


@app.route("/watchlist", methods=["GET"])

def get_watchlist():
    """Return the current user's watchlist symbols, with their last known price."""
    ensure_watchlist_table()
    email = _current_user_email()
    rows = lakebase.run_query(
        f"SELECT symbol, email, latest_price, updated_at FROM {WATCHLIST_TABLE_NAME} "
        f"WHERE email = %s ORDER BY symbol ASC",
        (email,),
    )
    return jsonify(rows)


@app.route("/watchlist", methods=["POST"])
def add_to_watchlist():
    """
    Fetch the latest price for a single stock symbol from Massive using
    exactly ONE API call (see MassiveClient.get_latest_price), then add/
    update that symbol on the watchlist in Lakebase.
    """
    ensure_watchlist_table()

    if request.is_json:
        symbol = request.json.get("symbol", "")
    else:
        symbol = request.form.get("symbol", "")

    symbol = symbol.strip().upper() if isinstance(symbol, str) else ""

    if not symbol or not _TICKER_RE.match(symbol):
        return jsonify({"error": f"Invalid ticker symbol: {symbol!r}"}), 400

    client = MassiveClient()
    try:
        data = client.get_latest_price(symbol)  # <-- single API call, latest price only
    except requests.HTTPError:
        # Massive returns a 404/4xx for tickers it doesn't recognize.
        return jsonify({"error": f"Unknown ticker symbol: {symbol}"}), 400

    price = _extract_latest_price(data)
    if price is None:
        # No usable price in the response (e.g. delisted/invalid ticker
        # that still 200s with an empty result set) - don't add it.
        return jsonify({"error": f"No price data available for ticker: {symbol}"}), 400

    email = _current_user_email()

    lakebase.run_write(
        f"""
        INSERT INTO {WATCHLIST_TABLE_NAME} (symbol, email, latest_price, updated_at)
        VALUES (%s, %s, %s, now())
        ON CONFLICT (symbol, email) DO UPDATE
            SET latest_price = EXCLUDED.latest_price,
                updated_at = EXCLUDED.updated_at
        """,
        (symbol, email, price),
    )

    return jsonify({"symbol": symbol, "email": email, "latest_price": price})


@app.route("/watchlist/<symbol>", methods=["DELETE"])
def delete_from_watchlist(symbol: str):
    """Remove a single symbol from the current user's watchlist."""
    ensure_watchlist_table()

    symbol = symbol.strip().upper() if isinstance(symbol, str) else ""
    if not symbol or not _TICKER_RE.match(symbol):
        return jsonify({"error": f"Invalid ticker symbol: {symbol!r}"}), 400

    email = _current_user_email()
    deleted = lakebase.run_write(
        f"DELETE FROM {WATCHLIST_TABLE_NAME} WHERE symbol = %s AND email = %s",
        (symbol, email),
    )

    if not deleted:
        return jsonify({"error": f"{symbol} is not on your watchlist"}), 404

    return jsonify({"symbol": symbol, "email": email, "deleted": True})


def _extract_latest_price(data: dict) -> float | None:
    """Pull the trade price out of the Massive 'previous close' response shape.

    The /v2/aggs/ticker/{symbol}/prev endpoint returns "results" as a LIST
    containing a single aggregate bar (not a dict), e.g.:
        {"status": "OK", "resultsCount": 1, "results": [{"c": 148.845, ...}]}
    Previously this code treated "results" as a dict, so isinstance(results, dict)
    was always False for this endpoint's real shape and the price silently
    resolved to None. Unwrap the list here, and check "status"/"resultsCount"
    so invalid tickers (empty results) are detected instead of "succeeding"
    with a null price.

    Adjust the key lookup here if the real Massive API returns a different
    field name for the traded/close price.
    """
    if not isinstance(data, dict):
        return None
    if data.get("status") not in (None, "OK") or data.get("resultsCount") == 0:
        return None
    results = data.get("results", data)
    if isinstance(results, list):
        results = results[0] if results else None
    if isinstance(results, dict):
        for key in ("c", "p", "price", "last_price", "vw"):
            if key in results:
                return results[key]
    return None


def _upsert_batch(items: list[dict]) -> int:
    """Upsert a batch of Massive API items into Lakebase, one statement per row.

    For very large batches, consider psycopg2.extras.execute_values for
    higher throughput instead of per-row execute calls.
    """
    import json as _json

    count = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for item in items:
                cur.execute(
                    f"""
                    INSERT INTO {TABLE_NAME} (id, payload, synced_at)
                    VALUES (%s, %s, now())
                    ON CONFLICT (id) DO UPDATE
                        SET payload = EXCLUDED.payload,
                            synced_at = EXCLUDED.synced_at
                    """,
                    (str(item.get("id")), _json.dumps(item)),
                )
                count += 1
            conn.commit()
    return count


def _upsert_news_batch(ticker: str, articles: list[dict]) -> int:
    """Upsert news articles for a single ticker into the news documents table.

    Flattens the top-level "insights" sentiment entry that matches this
    ticker (if present) into its own columns so the Spark notebook can read
    plain text columns instead of parsing JSONB for the common case.
    """
    import json as _json

    count = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for article in articles:
                sentiment = None
                sentiment_reasoning = None
                for insight in article.get("insights", []) or []:
                    if insight.get("ticker") == ticker:
                        sentiment = insight.get("sentiment")
                        sentiment_reasoning = insight.get("sentiment_reasoning")
                        break

                publisher = article.get("publisher") or {}
                cur.execute(
                    f"""
                    INSERT INTO {NEWS_TABLE_NAME} (
                        id, ticker, title, description, author, article_url,
                        publisher_name, keywords, sentiment, sentiment_reasoning,
                        published_utc, payload, synced_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (id) DO UPDATE
                        SET ticker = EXCLUDED.ticker,
                            title = EXCLUDED.title,
                            description = EXCLUDED.description,
                            author = EXCLUDED.author,
                            article_url = EXCLUDED.article_url,
                            publisher_name = EXCLUDED.publisher_name,
                            keywords = EXCLUDED.keywords,
                            sentiment = EXCLUDED.sentiment,
                            sentiment_reasoning = EXCLUDED.sentiment_reasoning,
                            published_utc = EXCLUDED.published_utc,
                            payload = EXCLUDED.payload,
                            synced_at = EXCLUDED.synced_at
                    """,
                    (
                        str(article.get("id")),
                        ticker,
                        article.get("title", ""),
                        article.get("description"),
                        article.get("author"),
                        article.get("article_url"),
                        publisher.get("name"),
                        _json.dumps(article.get("keywords", [])),
                        sentiment,
                        sentiment_reasoning,
                        article.get("published_utc"),
                        _json.dumps(article),
                    ),
                )
                count += 1
            conn.commit()
    return count


if __name__ == '__main__':
    host = os.getenv('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_RUN_PORT', 8000))
    app.run(debug=True, host=host, port=port)
    print(f"Flask app running on http://{host}:{port}")