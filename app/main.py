import logging
import os
import secrets
from time import monotonic
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.db import get_session
from app.embedding import (
    EmbeddingUnavailable,
    embed_client_descriptions,
    index_documents,
    upsert_vocab,
    vocab_for_clients,
    vocab_for_documents,
)
from app.models import Client, Document
from app.schemas import (
    ClientCreate,
    ClientHit,
    ClientResponse,
    DocumentCreate,
    DocumentHit,
    DocumentResponse,
)
from app.search import client_hits, document_hits

app = FastAPI(
    title="Nevis Search API",
    description="Hybrid client and document search API with normalized indexing, fuzzy typo recovery, and verbatim evidence highlighting.",
)
DatabaseSession = Annotated[Session, Depends(get_session)]
bearer_scheme = HTTPBearer(auto_error=False)


def require_api_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> None:
    """Enforce Bearer token auth when API_TOKEN is configured (e.g. cloud deployment)."""
    expected = os.getenv("API_TOKEN")
    if not expected:
        return
    if credentials is None or not secrets.compare_digest(credentials.credentials, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


Protected = [Depends(require_api_token)]


@app.exception_handler(EmbeddingUnavailable)
def embedding_error(request, error):
    return JSONResponse(status_code=503, content={"detail": str(error)})


@app.post("/clients", status_code=status.HTTP_201_CREATED, dependencies=Protected)
def create_client(payload: ClientCreate, session: DatabaseSession) -> ClientResponse:
    client = Client(**payload.model_dump())
    embed_client_descriptions([client])
    session.add(client)
    session.flush()
    upsert_vocab(session, vocab_for_clients([client]))
    response = ClientResponse.model_validate(client)
    session.commit()
    return response


@app.post(
    "/clients/{id}/documents",
    status_code=status.HTTP_201_CREATED,
    dependencies=Protected,
    responses={404: {"description": "Client not found"}},
)
def create_document(
    id: str, payload: DocumentCreate, session: DatabaseSession
) -> DocumentResponse:
    if session.get(Client, id) is None:
        raise HTTPException(status_code=404, detail="Client not found")

    document = Document(client_id=id, **payload.model_dump())
    session.add(document)
    session.flush()
    session.add_all(index_documents([document]))
    upsert_vocab(session, vocab_for_documents([document]))
    response = DocumentResponse.model_validate(document)
    session.commit()
    return response


@app.get(
    "/documents/{id}",
    summary="Fetch a document by ID",
    dependencies=Protected,
    responses={404: {"description": "Document not found"}},
)
def get_document(id: str, session: DatabaseSession) -> DocumentResponse:
    document = session.get(Document, id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentResponse.model_validate(document)


@app.get("/search", summary="Search clients and documents", dependencies=Protected)
def search(
    q: Annotated[str, Query(min_length=1, max_length=256, description="Search query")],
    session: DatabaseSession,
    type: Annotated[
        Literal["client", "document"] | None,
        Query(description="Optional filter to search only clients or only documents"),
    ] = None,
    limit: Annotated[
        int,
        Query(ge=1, le=25, description="Maximum hits per result type (default 10)"),
    ] = 10,
    include_content: Annotated[
        bool,
        Query(description="Include full raw document content in document hits"),
    ] = False,
) -> list[ClientHit | DocumentHit]:
    q = q.strip()
    if not q or "\x00" in q:
        raise HTTPException(status_code=422, detail="Enter a nonblank text query without NUL characters")

    start = monotonic()
    corrected_q = None
    query_vec = None
    documents: list[DocumentHit] = []
    clients: list[ClientHit] = []
    has_strong_client = False

    # Compute document query embedding once and share it with client description fallback if needed.
    if type != "client":
        documents, corrected_q, query_vec = document_hits(
            session, q, limit=limit, include_content=include_content
        )
    if type != "document":
        clients, has_strong_client = client_hits(
            session, q, limit=limit, query_vector=query_vec, corrected_q=corrected_q
        )

    logging.getLogger("uvicorn.error").info(
        "Search completed in %.1f ms with %d clients and %d documents",
        (monotonic() - start) * 1000, len(clients), len(documents),
    )

    # Clients lead when a strong client match exists; otherwise put documents first if documents matched.
    if has_strong_client or not documents:
        return [*clients, *documents]
    return [*documents, *clients]
