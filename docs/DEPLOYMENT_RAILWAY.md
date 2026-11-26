# Deploying Nevis Search API on Railway

The repository's [Dockerfile](../Dockerfile) is Railway-ready out of the box:
- Listens on Railway's injected `${PORT}` (`--port ${PORT:-8000}`).
- Bakes the quantized `BAAI/bge-small-en-v1.5` ONNX weights into the image during `docker build` (no runtime model downloads or API keys).
- Normalizes `postgres://` and `postgresql://` connection strings automatically to `postgresql+psycopg://` in [app/db.py](../app/db.py).
- Runs `alembic upgrade head && python -m app.backfill` automatically on startup before launching Uvicorn.

---

## Step-by-Step Deployment

### 1. Create a Railway Project with `pgvector`
1. Create a new project on [Railway](https://railway.app/).
2. Add a **pgvector** database from Railway's template marketplace (uses `pgvector/pgvector:pg17`, which includes `vector`, `pg_trgm`, `unaccent`, and `fuzzystrmatch`).
   > **Important:** Choose the **pgvector** template rather than vanilla PostgreSQL, because migration `0001` runs `CREATE EXTENSION IF NOT EXISTS vector`.

### 2. Deploy the API Service from GitHub
1. Click **+ New → GitHub Repo** and select `nikgorbachev/nevis-search-api`.
2. Railway automatically detects the root `Dockerfile`.
3. Under **Variables**, add:
   ```text
   DATABASE_URL=${{pgvector.DATABASE_URL}}
   ```
   *(Replace `pgvector` with the exact service name of your PostgreSQL service in Railway.)*
4. Under **Settings → Networking**, click **Generate Domain** to expose HTTPS on port `8000` (Railway injects `PORT` automatically).
5. Wait for the first build & deploy to finish. Open `https://<your-domain>.up.railway.app/docs` to verify Swagger UI loads cleanly.

### 3. Seed & Backfill the Demo Corpus on Railway
Once deployed, load the deterministic 120-client / 1,486-document synthetic demo corpus from your laptop using the **Public Database URL** (`DATABASE_PUBLIC_URL`) from your Railway Postgres service's **Variables** tab:

```sh
# 1. Seed 120 synthetic clients and 1,486 realistic documents (idempotent):
python3 scripts/seed_demo.py | psql "$RAILWAY_PUBLIC_DATABASE_URL" -v ON_ERROR_STOP=1

# 2. Trigger backfill (either via Railway CLI or by clicking "Redeploy" on the API service):
railway run python -m app.backfill
```

> **Tip:** Because the container entrypoint runs `python -m app.backfill` on startup, simply clicking **Redeploy** (or **Restart**) on the API service after step 1 will automatically index all 1,486 documents and 120 client descriptions (~2–3 minutes once; subsequent restarts take <2 seconds).

### 4. Verify the Live Deployment
Run the integration tests directly against your Railway URL:

```sh
NEVIS_TEST_URL=https://<your-domain>.up.railway.app python3 -m unittest discover -s tests -v
```

Test a few showcase queries in Swagger UI or via `curl`:

```sh
curl -sG "https://<your-domain>.up.railway.app/search" --data-urlencode 'q=NevisWealth' --data 'type=client'
curl -sG "https://<your-domain>.up.railway.app/search" --data-urlencode 'q=address proof' --data 'type=document'
curl -sG "https://<your-domain>.up.railway.app/search" --data-urlencode 'q=ZK-4729'
curl -sG "https://<your-domain>.up.railway.app/search" --data-urlencode 'q=adres proof' --data 'type=document'
```
