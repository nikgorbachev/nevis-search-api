# Nevis client and document search

Find clients by name or email and find evidence inside their documents. FastAPI serves the HTTP API. PostgreSQL handles literal, full-text, trigram, and vector search.

## Start the API

Run from the repository root with Docker Compose.

```sh
docker compose up --build
```

The first build downloads Python dependencies and a pinned, quantized English BGE-small model. All inference runs locally on CPU. No external API keys are needed. After PostgreSQL passes its health check, the API applies the migrations and indexes existing documents before serving requests. Open [Swagger UI](http://localhost:8000/docs) once Uvicorn starts.

## Seed sample data

With the stack running, use Python 3 to create a synthetic client and a plain-text utility bill over HTTP.

```sh
python3 scripts/seed.py
```

For the larger interview demo, load 100 synthetic clients and 1,593 documents into the named PostgreSQL volume. The bulk fixture uses stable IDs. Both commands can be repeated.

```sh
python3 scripts/seed_demo.py | docker compose exec -T db psql -X -U nevis -d nevis -v ON_ERROR_STOP=1
docker compose exec api python -m app.backfill
```

## Search examples

Open `GET /search` in Swagger or run these requests after loading the larger demo.

```sh
curl -sG localhost:8000/search --data-urlencode 'q=NevisWealth' --data 'type=client'
curl -sG localhost:8000/search --data-urlencode 'q=address proof' --data 'type=document'
curl -sG localhost:8000/search --data-urlencode 'q=ZK-4729'
curl -sG localhost:8000/search --data-urlencode 'q=2.5%'
curl -sG localhost:8000/search --data-urlencode 'q=QQ-0047' --data 'view=grouped'
```

The default response is an array of typed hits. Clients appear first, followed by documents. Each type has its own relevance order and a cap of ten. Use `type=client` or `type=document` for one section, or `view=grouped` for an object with `clients` and `documents` arrays. A missing form code returns `[]` by default or `{"clients":[],"documents":[]}` in the grouped view. A blank query returns HTTP 422.

For `NevisWealth`, the client result is Jordan Example, with `matched_field` set to `email`. For `address proof`, the first document is the November electricity bill. Its preview includes the service address. Searching for `ZK-4729` returns this complete response from the seeded database.

```json
[
  {
    "type": "document",
    "id": "9b93ac62-e499-5316-b32f-2e3a38533e21",
    "client_id": "7a69d696-860f-55ba-961d-73a7af2b639a",
    "client_name": "Elena Müller",
    "title": "Family transition memorandum",
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
    }
  }
]
```

## Try it in Swagger

1. Open `POST /clients`, select **Try it out**, and execute this request. Expect **201 Created** and copy the returned `id`.

   ```json
   {
     "first_name": "Jordan",
     "last_name": "Example",
     "email": "jordan@example.com",
     "description": "Synthetic client for local testing",
     "social_links": ["https://example.com/jordan"]
   }
   ```

2. Open `POST /clients/{id}/documents`, select **Try it out**, paste the client ID into the `id` path field, and execute this request. Expect **201 Created** with the same ID in `client_id` and a `created_at` timestamp.

   ```json
   {
     "title": "November electricity bill",
     "content": "Electricity utility bill\nJordan Example\n12 Sample Lane\nBilling period: November 2026\nAmount due: EUR 58.20"
   }
   ```

   Repeat with `missing-client` as the ID to get **404 Not Found**.

Check that PostgreSQL has enabled the extension.

```sh
docker compose exec db psql -U nevis -d nevis -c "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
```

Run the HTTP integration tests against the running stack.

```sh
python3 -m unittest discover -s tests -v
```

## How ranking works

- Clients use case-insensitive email lookup, accent-insensitive name matching, literal fragments and description search. A trigram fallback handles misspelled names. Input punctuation is literal throughout substring matching.
- Documents are indexed when created, in the same transaction as their passages. Short documents stay whole. Long ones split near line breaks around 180 tokenizer tokens, with 20 tokens of overlap for forced splits. The title supplies context for the local 384-dimensional embedding. The source content and character offsets support verbatim previews.
- Each document contributes its best passage to each shortlist. PostgreSQL ranks up to 20 unique documents by weighted English full-text matches and up to 20 by exact cosine similarity. Weighted reciprocal rank fusion adds `weight / (60 + rank)` per eligible arm. A known form code or percentage requires a literal match with token boundaries. Queries such as `address proof` have one explicit expansion to the vocabulary of a utility bill and service address. Without it, our local model favored onboarding checklists.
- Evidence contains the original passage excerpt, its zero-based start in the source title or content, and half-open highlight ranges relative to the excerpt. Offsets count Unicode code points. A semantic match may have an empty highlight list. Search timing and result counts are logged without the query text.
- The initial model, `BAAI/bge-small-en-v1.5`, runs through ONNX Runtime. Its quantized ONNX export is pinned by revision in the Dockerfile. The encoder uses CLS pooling and L2 normalization. Changing the embedding model requires rebuilding all passage vectors. An exact scan keeps this small corpus simple. Larger deployments could add indexes once latency warrants them. [Model card](https://huggingface.co/BAAI/bge-small-en-v1.5) · [ONNX export](https://huggingface.co/Xenova/bge-small-en-v1.5)

## Retrieval evaluation

With the demo indexed and the API running, replay the document queries, client lookups and evidence checks from the repository root.

```sh
docker compose exec -T api python - < scripts/evaluate.py
```

The development set contains six positive and three negative document queries. A cosine floor of 0.60 rejected all three negatives. With the address-proof expansion, fixed weights of 70% lexical and 30% semantic placed all six target titles first. Equal weights placed five first. The query-length schedule tied the fixed weights on this set.

On seven held-out positives and three held-out negatives, the title-level results were:

| Ranking | Target first | Target in top five | Negatives with hits |
| --- | ---: | ---: | ---: |
| Lexical | 4/7 | 5/7 | 0/3 |
| Semantic | 5/7 | 5/7 | 1/3 |
| Hybrid | 5/7 | 5/7 | 0/3 |

The fixed corpus in this run contained the 1,593 demo documents and two earlier example records. The labels accept any document with the expected title. Each arm uses the same domain expansion. The semantic-only comparison also tries vectors for identifier queries, which take the literal path in the shipped API.

Two held-out queries missed their labeled title. Asking where the household lives surfaced a storage inventory. Asking about conditions on payment after a company sale surfaced a family transition memo. The latter covers a related topic. These synthetic judgments cover a few representative tasks and leave room for better paraphrase handling. A shortlist reranker would need to show a gain on a larger labeled set to justify the extra inference.
