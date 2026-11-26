# Architecture, Retrieval Design & Product Tradeoffs

This document explains the engineering, information-retrieval, and product decisions behind **Nevis Search API** across three perspectives: **Data Science & Retrieval**, **Product & API Design**, and **Testing & Scalability**.

---

## 1. Data Science & Retrieval Architecture

### 1.1 Three-Step Document Retrieval Funnel
Advisors searching wealth-management records do not just need a document title — they need to verify the exact clause (a form code, fee rate, or service address) before opening a multi-page file. The system separates retrieval into three explicit stages:

```text
Query (q)
  │
  ├─► Normalization & Guarded Spell Correction (app/normalize.py, search_vocab + Damerau-Levenshtein)
  │     • Protects valid single-token English words via BGE's 30k WordPiece vocab
  │     • Never rewrites strict IDs or percentages (ZK-4728 stays strict negative)
  │
  ├─► Step 1: Parent Discovery & Step 2: Segment Retrieval (PostgreSQL)
  │     ├── Lexical Arm: precomputed weighted tsvector (GIN) + trigram LIKE (GIN) + quorum coverage gate
  │     └── Semantic Arm: BGE-small-en-v1.5 (ONNX CPU, 384-dim) + pgvector HNSW cosine shortlist
  │     └── Fusion: Weighted RRF (0.7 lex / 0.3 sem, k=60) + Two-Tier Semantic Floor (0.60 dual / 0.69 sem-only)
  │
  └─► Step 3: Cue Localization & Verbatim Evidence (app/search.py:evidence)
        • Picks highest query-term-density sentence/row inside winning chunk
        • Highlights exact, accent-folded, compact-ID, or stemmed tokens
        • Preserves exact raw Unicode character offsets (never hallucinates highlights on paraphrases)
```

### 1.2 Store Raw, Search Normalized, Display Clean
WealthTech platforms ingest dirty documents (OCR scans with broken line-wraps and ligatures, forwarded email threads with `> >` prefixes, spaced percentages like `1.5 %`).
- **Storage (`documents.content`)**: Kept **100% raw and immutable** for compliance/auditability and exact offset verification (`original[start : start + len(text)] == text`).
- **Indexing ([app/normalize.py](../app/normalize.py))**: Derives normalized text at ingest/backfill for embeddings, `search_vector` (`tsvector`), and trigram indexes:
  - NFKC normalization (`ﬁ` → `fi`, `ﬂ` → `fl`, smart quotes/NBSP cleanup)
  - PDF/OCR line-wrap de-hyphenation (`elec-\ntricity` → `electricity`)
  - Spaced OCR header repair (`U T I L I T Y` → `UTILITY`)
  - Email quote/divider stripping (`> >`, `-----Original Message-----`)
  - Percentage and currency spacing (`1.5 %` → `1.5%`, `EUR142.80` → `EUR 142.80`)
  - Compact form-code indexing (`ZK-4729` also indexed as `zk4729`, while keeping digit boundaries strict)
- **Display (`summary`)**: Generates a clean, readable 1–2 sentence extractive summary from the normalized text (stripping email headers), answering the assignment's optional *"Quick summary of document content"* with zero external LLM dependency.

### 1.3 OpenSearch-Style Fuzzy Search inside PostgreSQL
Pure `pg_trgm` similarity struggles on short words and transpositions (`Elna`, `Muler`, `Marn Red`, `cuorier`), while blind edit-distance rewriting can turn real English words into unrelated proper names (e.g. `"margin"` → client name `"Marin"`).

We solve both cleanly inside PostgreSQL (`fuzzystrmatch` + `pg_trgm`):
- **Clients**: Tiered exact/prefix/substring SQL runs first. When exact name/email lookup finds nothing, a token-level `levenshtein_less_equal` fallback (OpenSearch `AUTO` edit distance: `≤1` for 3–5 chars, `≤2` for 6+ chars) matches across name tokens and split email tokens (`Elna`, `Muler`, `Marn Red`, `neviswelth`, `welth`), followed by semantic vector search over `clients.description_embedding` (`philanthropy` → `"charitable commitment"`, `exiting a family company` → `"gradual exit from a family business"`).
- **Documents**: Out-of-vocabulary query tokens are checked against BGE's 30,522-word WordPiece vocabulary first. Valid single-token English words (`margin`, `liquidation`, `exiting`, `penguin`) are **never** rewritten. Multi-subword typos (`portfolo`, `cuorier`, `electrcity`, `adres`, `vestng`, `escow`) are resolved via Damerau-Levenshtein (treating adjacent transpositions like `cuorier` ↔ `courier` as distance `1`) against `search_vocab`.

### 1.4 Precision & True-Negative Gating (Why RRF Alone Is Not Enough)
RRF only orders candidates within each arm — it cannot tell whether *any* document is genuinely relevant. Without admission gates, out-of-corpus financial queries (`cryptocurrency staking yield`, `options Greeks delta hedging`, `Swiss numbered bank account IBAN`) previously produced 10 false positives each.

Two admission gates fix this:
1. **Lexical Minimum-Term Quorum Gate**: For queries with ≥3 lexemes (non-alias), require at least `max(2, ceil(0.45 * N))` distinct query lexemes (or an exact phrase hit) so a single generic word like `"notice"` or `"account"` cannot admit 20 unrelated documents.
2. **Two-Tier Semantic Floor**:
   - `COSINE_FLOOR = 0.60` when the lexical arm also admitted the document (dual-arm consensus).
   - `SEMANTIC_ONLY_FLOOR = 0.69` when semantic retrieval is acting alone (rejects near-domain financial negatives clustering around `0.60–0.66`).

### 1.5 Why Local Quantized BGE-Small (and Why No Reranker Yet)
- **Model choice**: Quantized `BAAI/bge-small-en-v1.5` (ONNX Runtime, 384-dim) is baked into the Docker image at build time (~33 MB quantized weights, ~11–20 ms per query on CPU). Reviewers need **zero API keys**, zero external network calls at runtime, and zero billing setup.
- **No cross-encoder reranker in MVP**: On this corpus, indexed hybrid RRF already achieves **0.971 Doc Hit@1** and **0.909 Dedup Seg NDCG@5** in **45.4 ms median**. A CPU cross-encoder adds ~300–1,500 ms and is only justified once a larger production query log shows relevant documents landing at ranks 2–5 that reranking can consistently promote without hurting exact identifier queries.

---

## 2. Product & API Design Tradeoffs

| Decision | Choice Shipped | Tradeoff & Rationale |
| :--- | :--- | :--- |
| **Response shape** | Single flat JSON array `list[ClientHit \| DocumentHit]` | Strictly matches the OpenAPI `type: array` contract in the brief. Avoids polymorphic `array \| object` responses that break typed SDK generation. Any frontend can group by `hit.type` in two lines. |
| **Optional `type` filter** | `type=client\|document` (default: both) | Mirrors Spotify-style "All / Clients / Documents" filter pills and skips document embeddings/pgvector entirely on client-only searches (~10–35 ms). |
| **Lean previews vs. full bodies** | Return `summary` + verbatim `evidence` by default; `include_content=true` & `GET /documents/{id}` for drill-down | Returning 10 multi-page documents on every search bloats payloads and defeats evidence previews. `ClientHit` returns the full `Client` schema (`description`, `social_links`) because client rows are small and needed to explain `matched_field: "description"`. |
| **Intent-aware ordering** | Clients lead when a strong client match (`email`/`name`) exists; otherwise Documents lead | Gives predictable mixed search (`NevisWealth` / `Elena Muller` lead with the client; `ZK-4729` / `address proof` lead with the document) without an opaque black-box cross-type score normalizer. |
| **No pagination** | Top-N relevance cap (`limit=1..25`, default `10` per type) | Search bars optimize for fast top-k precision rather than fabricating a total count from a bounded hybrid shortlist. Exhaustive browsing belongs on a cursor-paginated listing endpoint. |
| **Domain aliases** | Small explicit curated map (`address proof`, `kyc`, `poa`, `rmd`, `w-9`) | Transparent, easy to audit, and typo-tolerant (`adres proof` expands cleanly). Avoids over-engineering an opaque ontology before real advisor query logs exist. |

---

## 3. Testing, Evaluation & Scalability

### 3.1 Layered Test & Evaluation Strategy
- **Pure Unit Tests ([tests/test_normalize.py](../tests/test_normalize.py), [tests/test_ranking_unit.py](../tests/test_ranking_unit.py))**: Fast deterministic tests covering OCR/email normalization, summary generation, Damerau-Levenshtein transpositions, alias expansion, RRF two-tier semantic gating, density sentence selection, accent-folded highlighting, and zero hallucinated highlights.
- **Self-Contained HTTP Integration Tests ([tests/test_api.py](../tests/test_api.py), [tests/test_search.py](../tests/test_search.py))**: Run cleanly against **both** a freshly booted empty database and a seeded demo database (using unique per-run fixture tokens).
- **3-Fold Offline + Live Evaluation ([scripts/evaluate.py](../scripts/evaluate.py))**: Measures `Doc Hit@1/5`, `MRR`, `Segment Hit@1`, `Deduplicated Graded Segment NDCG@5`, `Cue Highlight Exactness`, `Offset Integrity`, `True Negative Rate` across 4 negative subclasses, and stage-by-stage latency.

### 3.2 Indexing & Firm-Scale Readiness (120 vs. 1,000 Clients)
Initial profiling revealed that **85% of warm search latency (~360–412 ms)** came from recomputing `to_tsvector()` on the fly — while ONNX embedding (~15 ms) and pgvector (~20 ms) were fast.

Migration [0003_search_upgrades.py](../migrations/versions/0003_search_upgrades.py) adds:
- Precomputed `document_chunks.search_vector` (`GIN`)
- `pg_trgm` `GIN` indexes on normalized chunk titles/passages, `search_vocab`, and `clients`
- `pgvector` `HNSW` index (`vector_cosine_ops`) on `document_chunks.embedding`

Result: warm document search HTTP median dropped from **350 ms → 45.4 ms (7.7× speedup)**, and `scripts/seed_demo.py --clients 1000` + parallel `python -m app.backfill` (`FOR UPDATE SKIP LOCKED`) scales cleanly to firm-sized corpora (1,000+ clients / 15,000+ documents) without changing query architecture.
