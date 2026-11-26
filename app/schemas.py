from typing import Literal
from pydantic import AwareDatetime, BaseModel, ConfigDict, EmailStr, Field


class ClientCreate(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    description: str | None = None
    social_links: list[str] = Field(default_factory=list)


class ClientResponse(ClientCreate):
    model_config = ConfigDict(from_attributes=True)

    id: str


class DocumentCreate(BaseModel):
    title: str
    content: str


class DocumentResponse(DocumentCreate):
    model_config = ConfigDict(from_attributes=True)

    id: str
    client_id: str
    created_at: AwareDatetime


class ClientHit(BaseModel):
    type: Literal["client"] = "client"
    id: str
    first_name: str
    last_name: str
    email: str
    matched_field: str


class Span(BaseModel):
    start: int
    end: int


class Evidence(BaseModel):
    source: Literal["title", "content"]
    start: int
    text: str
    highlights: list[Span]


class DocumentHit(BaseModel):
    type: Literal["document"] = "document"
    id: str
    client_id: str
    client_name: str
    title: str
    evidence: Evidence


class SearchResponse(BaseModel):
    clients: list[ClientHit]
    documents: list[DocumentHit]
