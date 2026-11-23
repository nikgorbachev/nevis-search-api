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
