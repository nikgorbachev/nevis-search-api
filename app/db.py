import os
from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session


def _database_url() -> str:
    url = os.environ["DATABASE_URL"]
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url.removeprefix("postgres://")
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url.removeprefix("postgresql://")
    return url


engine = create_engine(_database_url(), pool_pre_ping=True, hide_parameters=True)


def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session
