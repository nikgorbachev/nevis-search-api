import logging
from time import monotonic
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.db import get_session
from app.embedding import EmbeddingUnavailable, index_documents
from app.search import client_hits, document_hits
from app.models import Client, Document
from app.schemas import (
    ClientCreate, ClientHit, ClientResponse, DocumentCreate, DocumentHit, DocumentResponse, SearchResponse,
)

app = FastAPI(title="Nevis Search API")
DatabaseSession = Annotated[Session, Depends(get_session)]


@app.exception_handler(EmbeddingUnavailable)
def embedding_error(request, error):
    return JSONResponse(status_code=503, content={"detail": str(error)})


@app.post("/clients", status_code=status.HTTP_201_CREATED)
def create_client(payload: ClientCreate, session: DatabaseSession) -> ClientResponse:
    client = Client(**payload.model_dump())
    session.add(client)
    session.flush()
    response = ClientResponse.model_validate(client)
    session.commit()
    return response


@app.post(
    "/clients/{id}/documents",
    status_code=status.HTTP_201_CREATED,
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
    response = DocumentResponse.model_validate(document)
    session.commit()
    return response


@app.get("/search", summary="Search clients and documents")
def search(
    q: Annotated[str, Query(min_length=1, max_length=256)],
    session: DatabaseSession,
    type: Literal["client", "document"] | None = None,
    view: Literal["list", "grouped"] = "list",
) -> list[ClientHit | DocumentHit] | SearchResponse:
    q = q.strip()
    if not q or "\x00" in q:
        raise HTTPException(status_code=422, detail="Enter a nonblank text query without NUL characters")
    start = monotonic()
    clients = client_hits(session, q) if type != "document" else []
    documents = document_hits(session, q) if type != "client" else []
    logging.getLogger("uvicorn.error").info(
        "Search completed in %.1f ms with %d clients and %d documents",
        (monotonic() - start) * 1000, len(clients), len(documents),
    )
    results = SearchResponse(clients=clients, documents=documents)
    if view == "grouped":
        return results
    return [*results.clients, *results.documents]
