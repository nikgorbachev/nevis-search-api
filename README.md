# nevis-search-api

## Start the API

Run from the repository root with Docker Compose.

```sh
docker compose up --build
```

The API applies the Alembic migration after PostgreSQL passes its health check. No external API keys are needed. Open [Swagger UI](http://localhost:8000/docs) once Uvicorn starts.

## Seed sample data

With the stack running, use Python 3 to create a synthetic client and a plain-text utility bill over HTTP.

```sh
python3 scripts/seed.py
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
