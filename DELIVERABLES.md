# Nevis Backend Home Assignment — Submission Deliverables

**Author:** Nikolai Gorbachev  
**Repository:** https://github.com/nikgorbachev/nevis-search-api  
**Live Deployed API (Railway):** https://nevis-search-api-production-04d73498.up.railway.app/docs *(pre-seeded with 120 synthetic clients & 1,486 documents; protected via Bearer `API_TOKEN` provided in the submission notes)*  
**Tech Stack:** Python 3.13, FastAPI, SQLAlchemy 2.0, Alembic, PostgreSQL 17 (`pgvector`, `pg_trgm`, `unaccent`, `fuzzystrmatch`), Local Quantized ONNX `BAAI/bge-small-en-v1.5` (CPU, key-free)

---

## Checklist Against Assignment Deliverables

| Deliverable | Link / Location | Notes |
| :--- | :--- | :--- |
| **Source code in a repo** | https://github.com/nikgorbachev/nevis-search-api | Clean, self-contained FastAPI + PostgreSQL implementation (`app/`, `migrations/`, `scripts/`, `tests/`). |
| **Docker Compose for reproducibility** | [compose.yaml](https://github.com/nikgorbachev/nevis-search-api/blob/main/compose.yaml) · [Dockerfile](https://github.com/nikgorbachev/nevis-search-api/blob/main/Dockerfile) | Single command `docker compose up --build` starts PostgreSQL + pgvector, applies migrations, backfills embeddings/indexes, and serves Swagger on `http://localhost:8000/docs`. Zero external API keys required. |
| **Tests for core logic and edge cases** | [tests/](https://github.com/nikgorbachev/nevis-search-api/tree/main/tests) | Pure unit tests (`test_normalize.py`, `test_ranking_unit.py`) + self-contained HTTP integration tests (`test_api.py`, `test_search.py`) covering exact/typo/semantic search, dirty OCR normalization, `%_` escaping, percentage boundaries, near-domain negatives, and HTTP `404`/`422` validation. |
| **README with setup & search examples** | [README.md](https://github.com/nikgorbachev/nevis-search-api/blob/main/README.md) | Step-by-step Docker setup, seeding commands, `curl` + JSON response examples, and 3-fold retrieval evaluation summary. |
| **API documentation (Swagger & Markdown)** | [Live Swagger UI](https://nevis-search-api-production-04d73498.up.railway.app/docs) · `http://localhost:8000/docs` · [docs/ARCHITECTURE.md](https://github.com/nikgorbachev/nevis-search-api/blob/main/docs/ARCHITECTURE.md) | OpenAPI 3.1 interactive Swagger docs + deep-dive engineering/product/retrieval design document. |
| **Optional: Quick summary of document content** | [app/normalize.py](https://github.com/nikgorbachev/nevis-search-api/blob/main/app/normalize.py) (`summarize_document`) | Every `DocumentHit` includes a clean extractive `summary` alongside verbatim `evidence` offsets and highlight spans. |
| **Optional: Deployed API (Railway)** | [Live API](https://nevis-search-api-production-04d73498.up.railway.app/docs) · [docs/DEPLOYMENT_RAILWAY.md](https://github.com/nikgorbachev/nevis-search-api/blob/main/docs/DEPLOYMENT_RAILWAY.md) | Live public HTTPS deployment on Railway + `pgvector` protected by Bearer `API_TOKEN` (open in Swagger UI via **Authorize** button; synthetic demo data only), verified end-to-end with 17/17 showcase checks and integration tests (`186 ms` median RTT). |

---

## Quick Start for Reviewers

```sh
git clone https://github.com/nikgorbachev/nevis-search-api.git
cd nevis-search-api
docker compose up --build
```

In a second terminal (once Uvicorn starts):

```sh
# Load the 120-client / 1,486-document synthetic wealth-management demo corpus:
python3 scripts/seed_demo.py | docker compose exec -T db psql -X -U nevis -d nevis -v ON_ERROR_STOP=1
docker compose exec api python -m app.backfill

# Run unit & HTTP integration tests:
python3 -m unittest discover -s tests -v
docker compose exec -T api python -m unittest discover -s tests -v

# Run the 3-fold retrieval evaluation (Doc Hit@1/5, Seg NDCG@5, Cue Highlights, Negatives, Latency):
docker compose exec -T api python - < scripts/evaluate.py
```

---

## Highlights & Measured Quality

- **Assignment core cases verified:**
  - `GET /search?q=NevisWealth&type=client` → returns `Jordan Example` (`jordan.doe@neviswealth.example.com`, `matched_field: "email"`).
  - `GET /search?q=address+proof&type=document` → ranks `November electricity bill` #1 with verbatim service-address evidence and clean summary.
  - `GET /search?q=ZK-4729` (and compact `ZK4729`) → locates the exact clause inside a multi-section memo with exact Unicode highlight offsets (`ZK-4728` returns `[]`).
  - `GET /search?q=2.5%` vs `25%` → strict numeric boundary isolation.
- **Normalization + OpenSearch-style fuzzy search:**
  - Store raw (`documents.content` immutable), search normalized (NFKC ligatures, OCR hyphenation/spacing, email noise), display clean (`summary`) with verifiable raw `evidence` offsets.
  - Token-level PostgreSQL `levenshtein_less_equal` + Damerau-Levenshtein transpositions + WordPiece single-token OOV guard recover typos (`welth`, `neviswelth`, `Elna`, `Muler`, `Marn Red`, `portfolo`, `cuorier`, `adres proof`) without false rewrites on real English words.
- **Measured 3-fold benchmark (120 clients / 1,486 docs / 26 multi-chunk docs):**
  - **Doc Hit@1 / Seg Hit@1 (all 35 queries incl. typos):** `0.971` (34/35) | **Dedup Seg NDCG@5:** `0.909`
  - **Typo subset (7 doc queries):** `7/7` (1.000) Doc & Seg Hit@1 | **Client suite (exact + typos + semantic desc):** `12/12`
  - **Cue Highlight Exactness:** `30/30` (100%) | **Verbatim Offset Integrity:** `34/34` (100%)
  - **True Negative Rate (14 negatives incl. near-domain finance & SQL wildcards):** `13/14` (92.9%)
  - **Warm HTTP Document Search Median:** `45.4 ms` (stored `tsvector` GIN + `pg_trgm` GIN + `pgvector` HNSW)
