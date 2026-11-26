# Nevis client and document search

Find clients by name, email, or description and locate verbatim evidence inside their documents. FastAPI serves the HTTP API. PostgreSQL handles normalized full-text search (GIN), Damerau-Levenshtein & trigram fuzzy matching (`fuzzystrmatch` + `pg_trgm`), and local semantic vector retrieval (`pgvector` HNSW).

> **Live Demo (Railway):** [Swagger UI](https://nevis-search-api-production-04d73498.up.railway.app/docs) (`https://nevis-search-api-production-04d73498.up.railway.app`, pre-seeded with 120 synthetic clients & 1,486 documents — no credentials required).

## Start the API

Run from the repository root with Docker Compose.

```sh
docker compose up --build
```

The first build downloads Python dependencies and a pinned, quantized English `BAAI/bge-small-en-v1.5` ONNX model. All inference runs locally on CPU with no external API keys. After PostgreSQL passes its health check, the API applies Alembic migrations (`0001`–`0003`) and backfills any unindexed documents or client descriptions before serving requests. Open [Swagger UI](http://localhost:8000/docs) once Uvicorn starts.

## Seed sample data

With the stack running, create a single synthetic client and utility bill over HTTP:

```sh
python3 scripts/seed.py
```

For the full wealth-management demo (120 synthetic clients, 1,486 documents including multi-section Investment Policy Statements, an Irrevocable Trust Deed, messy OCR scans, and forwarded email threads), load the deterministic fixture and run the repeatable backfill:

```sh
python3 scripts/seed_demo.py | docker compose exec -T db psql -X -U nevis -d nevis -v ON_ERROR_STOP=1
docker compose exec api python -m app.backfill
```

To benchmark firm-scale ingestion (e.g. 1,000 clients), pass `--clients 1000` to `scripts/seed_demo.py`. Multiple backfill workers can run safely in parallel (`FOR UPDATE SKIP LOCKED`).

## Search examples

Open `GET /search` in Swagger or run these requests after loading the demo corpus:

```sh
curl -sG localhost:8000/search --data-urlencode 'q=NevisWealth' --data 'type=client'
curl -sG localhost:8000/search --data-urlencode 'q=neviswelth' --data 'type=client'
curl -sG localhost:8000/search --data-urlencode 'q=address proof' --data 'type=document'
curl -sG localhost:8000/search --data-urlencode 'q=adres proof' --data 'type=document'
curl -sG localhost:8000/search --data-urlencode 'q=ZK-4729'
curl -sG localhost:8000/search --data-urlencode 'q=2.5%'
curl -sG localhost:8000/search --data-urlencode 'q=cryptocurrency staking yield' --data 'type=document'
```

### API contract

- `GET /search` always returns a **flat JSON array of typed hits** (`type: "client"` or `type: "document"`), matching the OpenAPI specification.
- Optional query parameters:
  - `type=client|document` — filter to one entity type (avoids document retrieval/embedding work on client-only searches).
  - `limit=1..25` — cap results per type (default `10`).
  - `include_content=true|false` — include full raw `content` on document hits (default `false`; drill down anytime via `GET /documents/{id}`).
- `ClientHit` returns the full `Client` fields (`id`, `first_name`, `last_name`, `email`, `description`, `social_links`) plus `type` and `matched_field`.
- `DocumentHit` returns `type`, `id`, `client_id`, `client_name`, `title`, `created_at`, a clean extractive `summary`, and a verbatim `evidence` preview anchored to raw `content` (or `title`) with exact Unicode character offsets and highlight spans.
- Strong exact/prefix/substring client matches lead mixed results; otherwise document matches appear first. Empty matches return `[]` with HTTP `200`; blank, over-length, or NUL queries return HTTP `422`.

Example response for `GET /search?q=ZK-4729`:

```json
[
  {
    "type": "document",
    "id": "9b93ac62-e499-5316-b32f-2e3a38533e21",
    "client_id": "7a69d696-860f-55ba-961d-73a7af2b639a",
    "client_name": "Elena Müller",
    "title": "Family transition memorandum",
    "created_at": "2026-11-26T03:35:19.409442Z",
    "summary": "Engagement scope The family asked for a single written record of decisions made during the first two planning meetings.",
    "evidence": {
      "source": "content",
      "start": 932,
      "text": "Form ZK-4729 must accompany the transfer to the successor custodian.",
      "highlights": [
        {
          "start": 5,
          "end": 12
        }
      ]
    },
    "content": null
  }
]
```

## Try it in Swagger

1. Open `POST /clients`, select **Try it out**, and execute:

   ```json
   {
     "first_name": "Jordan",
     "last_name": "Example",
     "email": "jordan@example.com",
     "description": "Synthetic client for local testing",
     "social_links": ["https://example.com/jordan"]
   }
   ```

   Expect **201 Created** and copy the returned `id`.

2. Open `POST /clients/{id}/documents`, paste the client ID into `id`, and execute:

   ```json
   {
     "title": "November electricity bill",
     "content": "Electricity utility bill\nJordan Example\n12 Sample Lane\nBilling period: November 2026\nAmount due: EUR 58.20"
   }
   ```

   Expect **201 Created**. Repeat with `missing-client` to get **404 Not Found**. Fetch any document directly via `GET /documents/{id}`.

Check PostgreSQL extensions and run both host and in-container unit/integration tests against the running stack:

```sh
docker compose exec db psql -U nevis -d nevis -c "SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'pg_trgm', 'unaccent', 'fuzzystrmatch') ORDER BY 1;"
python3 -m unittest discover -s tests -v
docker compose exec -T api python -m unittest discover -s tests -v
```

For deep-dive retrieval, product-tradeoff, and cloud deployment documentation, see:
- [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) — 3-step retrieval design, normalization layer, fuzzy matching, product tradeoffs, and scaling analysis
- [docs/DEPLOYMENT_RAILWAY.md](./docs/DEPLOYMENT_RAILWAY.md) — Railway + pgvector deployment and seeding guide
- [DELIVERABLES.md](./DELIVERABLES.md) — Reviewer quick-reference mapped to the assignment checklist

## Architecture & Ranking Tradeoffs

- **Store raw, search normalized, display clean:** Raw `documents.content` is never mutated. At index time, [app/normalize.py](./app/normalize.py) builds a conservative normalized view (NFKC ligature repair `ﬁ`→`fi`, PDF/OCR line-wrap de-hyphenation `elec-\ntricity`→`electricity`, spaced OCR header repair `U T I L I T Y`→`UTILITY`, email quote/divider cleanup, `%`/currency spacing, accent folding, and compact form-code aliases like `ZK4729`↔`ZK-4729`) for embeddings, precomputed weighted `tsvector` (`GIN`), and `pg_trgm` indexes. `summary` uses the cleaned text while `evidence` keeps exact raw Unicode character offsets.
- **Clients:** Tiered SQL priority (exact email → exact accent-folded name → prefix → substring name → substring email → description FTS). When exact name/email lookup misses, an OpenSearch-style token-level `levenshtein_less_equal` fuzzy fallback (`fuzzystrmatch`) recovers single- and multi-token name/email typos (`Elna`, `Muler`, `Marn Red`, `neviswelth`, `welth`), followed by a semantic description fallback over `clients.description_embedding` (`philanthropy` → charitable commitment, `exiting a family company` → gradual exit from a family business).
- **Documents (3-Step Funnel):**
  1. **Discovery & Chunking:** Short docs stay whole; long docs split near line breaks around 180 tokens (20-token overlap on forced splits) with title context baked into each 384-dim `BAAI/bge-small-en-v1.5` embedding (quantized ONNX, CLS pooling, L2 normalized).
  2. **Indexed Hybrid Retrieval + RRF:** Precomputed `search_vector` (`GIN`) + trigram (`GIN`) lexical shortlist (with a minimum query-term quorum gate so a single generic word cannot flood results) is fused with a `pgvector` `HNSW` cosine shortlist (`0.7` lexical / `0.3` semantic, `k=60`). Two-tier semantic admission (`0.60` when lexical agrees, `0.69` for semantic-only hits) rejects out-of-domain and near-domain false positives.
  3. **Typo Recovery & Verbatim Evidence:** Out-of-vocabulary subword-split query terms (never valid single-token English words in BGE's 30k vocabulary, and never numeric/form IDs) undergo Damerau-Levenshtein spell correction against `search_vocab` (`portfolo`→`portfolio`, `cuorier`→`courier`, `adres proof`→`address proof`), while strict identifier queries (`ZK-4729`, `ZK4729`, `2.5%`, `25%`) enforce exact token boundaries (`ZK-4728` returns `[]`). `evidence()` selects the highest-density matching sentence/row and highlights exact/accent-folded/stemmed tokens without fabricating highlights on paraphrases.

## 3-Fold Retrieval Evaluation

With the demo corpus indexed and the API running, run the QAD-2-inspired 3-fold evaluation suite (`Doc Hit@1/5`, `MRR`, `Segment Hit@1`, `Deduplicated Graded Segment NDCG@5`, `Cue Highlight Exactness`, `Offset Integrity`, `Negatives by Subclass`, and latency breakdown):

```sh
docker compose exec -T api python - < scripts/evaluate.py
```

Measured on the 120-client / 1,486-document corpus (1,536 chunks, 26 multi-chunk documents, `COSINE_FLOOR=0.60`, `SEMANTIC_ONLY_FLOOR=0.69`):

| Mode (`w_sem`) | Split | Doc Hit@1 | Doc Hit@5 | MRR | Seg Hit@1 | Dedup Seg NDCG@5 | Cue HL Exact | Offset Integrity |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Lexical-only (`0.0`) | All + Typos (35) | 0.943 | 0.943 | 0.943 | 0.943 | 0.895 | 30/30 | 34/34 |
| Semantic-only (`1.0`) | All + Typos (35) | 0.629 | 0.714 | 0.671 | 0.629 | 0.626 | 20/22 | 26/26 |
| **Hybrid shipped (`0.3`)** | **Non-Typo (28)** | **0.964** | **0.964** | **0.964** | **0.964** | **0.919** | **24/24** | **27/27** |
| **Hybrid shipped (`0.3`)** | **All + Typos (35)** | **0.971** | **0.971** | **0.971** | **0.971** | **0.909** | **30/30** | **34/34** |
| Hybrid equal (`0.5`) | All + Typos (35) | 0.914 | 0.971 | 0.943 | 0.914 | 0.885 | 30/30 | 34/34 |

- **Typo subset (7 doc queries: `portfolo`, `electrcity bill`, `adres proof`, `vestng shares`, `cuorier tracking`, `ZK4729`, `indemnity escow`):** **7/7 (1.000) Doc & Seg Hit@1**, **0.869 Seg NDCG@5**, **6/6 Cue Highlights**.
- **Client checks (exact, accent-folded, single/multi-token name & email typos, semantic descriptions):** **12/12**.
- **Negatives (14 queries across OOD, missing/near-miss IDs like `ZK-4728`, SQL wildcards `%_`/`%%`, and near-domain financial topics):** **13/14 True Negative Rate (92.9%)** — `0/3` OOD, `0/5` missing IDs, `0/2` wildcards, `1/4` near-domain (`margin call liquidation notice` matched the corpus's `Capital call notice CAP-3092` via `"call"` + `"notice"`).
- **Warm document search HTTP latency:** **45.4 ms median** (down from ~350 ms prior to stored `tsvector` GIN + HNSW indexing).
