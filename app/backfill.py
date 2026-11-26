"""Index existing documents, committing each batch atomically."""

from sqlalchemy import exists, func, select
from sqlalchemy.orm import Session

from app.db import engine
from app.embedding import index_documents
from app.models import Document, DocumentChunk


def main():
    missing = ~exists().where(DocumentChunk.document_id == Document.id)
    processed = 0
    while True:
        with Session(engine) as session, session.begin():
            # A second backfill process can safely take a different batch.
            documents = list(session.scalars(
                select(Document).where(missing).order_by(Document.id)
                .limit(32).with_for_update(skip_locked=True)
            ))
            if not documents:
                break
            session.add_all(index_documents(documents))
        processed += len(documents)
        if processed % 128 == 0:
            print(f"Indexed {processed} documents", flush=True)
    with Session(engine) as session:
        remaining = session.scalar(select(func.count()).select_from(Document).where(missing))
    print(f"Indexed {processed} documents. Remaining: {remaining}", flush=True)


if __name__ == "__main__":
    main()
