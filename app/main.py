from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from sqlalchemy.orm import Session

from app.db import get_session
from app.models import Client, Document
from app.schemas import ClientCreate, ClientResponse, DocumentCreate, DocumentResponse

app = FastAPI(title="Nevis Search API")
DatabaseSession = Annotated[Session, Depends(get_session)]


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
    response = DocumentResponse.model_validate(document)
    session.commit()
    return response
