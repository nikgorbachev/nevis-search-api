from datetime import datetime
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, DateTime, ForeignKey, Integer, Text, func
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def new_id() -> str:
    return str(uuid4())


class Base(DeclarativeBase):
    pass


class Client(Base):
    __tablename__ = "clients"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    first_name: Mapped[str] = mapped_column(Text)
    last_name: Mapped[str] = mapped_column(Text)
    email: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    social_links: Mapped[list[str]] = mapped_column(JSON)
    description_embedding: Mapped[list[float] | None] = mapped_column(Vector(384), nullable=True)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=new_id)
    client_id: Mapped[str] = mapped_column(Text, ForeignKey("clients.id"))
    title: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True)
    ordinal: Mapped[int] = mapped_column(primary_key=True)
    start_char: Mapped[int]
    end_char: Mapped[int]
    normalized_title: Mapped[str] = mapped_column(Text, default="")
    normalized_passage: Mapped[str] = mapped_column(Text, default="")
    search_vector: Mapped[str | None] = mapped_column(TSVECTOR, nullable=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(384))


class SearchVocab(Base):
    __tablename__ = "search_vocab"

    word: Mapped[str] = mapped_column(Text, primary_key=True)
    doc_freq: Mapped[int] = mapped_column(Integer, default=1)
