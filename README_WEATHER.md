# Vector Weather Retrieval Service — Weather Intelligence Pipeline

This directory contains the complete end-to-end implementation for the **Vector Weather Retrieval Service** built on **Databricks Apps** and **Lakebase** (Databricks-managed PostgreSQL with `pgvector`).

---

## 1. Data Source Selection & Rationale

We selected the **National Weather Service (NWS) API** (`https://api.weather.gov`) as our unstructured weather data source.

### Key API Endpoints Used:
1. `GET /alerts/active?area={state}`: Fetches active weather advisories, watches, and warnings for US state codes (e.g. `IL`, `TX`, `CA`).
   - Extracts unstructured narrative text from `description` and `instruction` fields.
2. `GET /points/{lat},{lon}` -> `GET /gridpoints/{office}/{x},{y}/forecast`: Resolves coordinates to grid offices and retrieves narrative multi-day forecast discussions (`detailedForecast` field).

### Rationale:
- **Rich Unstructured Narrative**: NWS provides natural-language prose describing severe weather risks, impacts, and safety guidance (ideal for dense vector embedding).
- **Public & Free**: No API keys required, high rate limits, and standard geo-location JSON formatting.

---

## 2. Database Schema & Architecture

The solution uses two primary tables created in Lakebase PostgreSQL with `pgvector`:

### `weather_documents` (Raw Document Store)
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `id` | `TEXT PRIMARY KEY` | Stable dedup key (`alert_id` or hash of `location + period + time`) |
| `location` | `TEXT NOT NULL` | Target region (state, city, or lat/lon pair) |
| `source_type` | `TEXT NOT NULL` | `'alert'` or `'forecast'` |
| `headline` | `TEXT` | Brief title or alert event name |
| `narrative_text` | `TEXT NOT NULL` | Full unstructured free-text narrative body |
| `issued_at` | `TIMESTAMPTZ` | Publication / effective timestamp |
| `payload` | `JSONB NOT NULL` | Raw JSON payload for auditability and lineage |
| `synced_at` | `TIMESTAMPTZ` | Timestamp of Lakebase ingestion |

### `weather_embeddings` (Vector Embeddings Store)
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `id` | `TEXT PRIMARY KEY` | Composite chunk ID (`{doc_id}_{chunk_index}`) |
| `document_id` | `TEXT NOT NULL` | Foreign key referencing `weather_documents(id)` |
| `chunk_index` | `INT NOT NULL` | Sequential chunk index within the document |
| `chunk_text` | `TEXT NOT NULL` | Unstructured text chunk content |
| `embedding` | `vector(384)` | 384-dimensional dense vector representation |
| `model_name` | `TEXT NOT NULL` | Transformer model tag (`all-MiniLM-L6-v2`) |
| `created_at` | `TIMESTAMPTZ` | Embedding generation timestamp |

### Indexing:
- **HNSW Index**: `CREATE INDEX idx_weather_embeddings_hnsw ON weather_embeddings USING hnsw (embedding vector_cosine_ops);`
  - Provides sub-linear execution time for cosine similarity search over large vector spaces.

---

## 3. Vectorization & Chunking Parameters

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
  - Selected to maintain complete compatibility and queryability with existing news pipelines in the app.
- **Chunking Strategy**: Sliding-window character chunker.
  - `CHUNK_SIZE`: 800 characters (~120–150 words).
  - `CHUNK_OVERLAP`: 100 characters (ensures critical context isn't severed across chunk boundaries).

---

## 4. REST API & Endpoints

### 1. Ingest / Sync Weather Data
`POST /weather/sync`
- **Request Body** (optional):
  ```json
  {
    "locations": ["IL", "TX", "Chicago, IL", "Austin, TX"],
    "limit": 50
  }
  ```
- **Response**:
  ```json
  {
    "synced": 37,
    "locations": ["IL", "TX", "Chicago, IL", "Austin, TX"],
    "total_harvested": 37
  }
  ```

### 2. Semantic Vector Retrieval
`POST /weather/search` or `GET /weather/search?query=...`
- **Request Body** (POST):
  ```json
  {
    "query": "flash flood risk near rivers",
    "top_k": 5,
    "source_type": "alert",
    "rag": true
  }
  ```
- **Response**:
  ```json
  {
    "query": "flash flood risk near rivers",
    "top_k": 5,
    "count": 3,
    "results": [
      {
        "id": "urn:oid:2.49.0.1.840.0...",
        "location": "Cook; DuPage; Will",
        "source_type": "alert",
        "headline": "Flash Flood Watch",
        "narrative_text": "Heavy rainfall expected...",
        "chunk_text": "Rapid rising water along small creeks and rivers...",
        "similarity": 0.8412
      }
    ],
    "rag_summary": "Summary for 'flash flood risk near rivers': High relevance report (Flash Flood Watch) found for location 'Cook; DuPage; Will'..."
  }
  ```

---

## 5. End-to-End Execution Guide

### Local Execution & Testing:
1. Ensure `.env` contains your `LAKEBASE_URL`.
2. Sync weather data from NWS API into Lakebase:
   ```bash
   python -c "import requests; print(requests.post('http://localhost:8000/weather/sync', json={'locations': ['IL', 'TX']}).json())"
   ```
3. Run the vector embedding pipeline manually if desired:
   ```bash
   python notebooks/ingest_weather_embeddings.py
   ```
4. Query the vector retrieval REST API:
   ```bash
   python -c "import requests; print(requests.post('http://localhost:8000/weather/search', json={'query': 'extreme heat warning', 'top_k': 3}).json())"
   ```

### Databricks Deployment:
1. Deploy as a Databricks App via workspace UI using `app.yaml`.
2. Schedule embedding updates with Databricks Asset Bundles:
   ```bash
   databricks bundle deploy -t dev
   databricks bundle run ingest_weather_embeddings_job -t dev
   ```

---

## 6. Known Limitations & Future Enhancements

- **Geocoding Expansion**: Future iterations can integrate OpenStreetMap Nominatim for arbitrary free-form location geocoding.
- **Dynamic HyDE (Hypothetical Document Embeddings)**: Expanding short search queries into detailed hypothetical weather bulletins prior to vector lookup.
