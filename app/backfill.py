"""Index documents, normalized search vectors, vocabulary, and client description embeddings atomically."""

from sqlalchemy import delete, exists, func, or_, select
from sqlalchemy.orm import Session

from app.db import engine
from app.embedding import (
    embed_client_descriptions,
    index_documents,
    upsert_vocab,
    vocab_for_clients,
    vocab_for_documents,
)
from app.models import Client, Document, DocumentChunk


def main():
    needs_index = or_(
        ~exists().where(DocumentChunk.document_id == Document.id),
        exists().where(
            DocumentChunk.document_id == Document.id,
            DocumentChunk.search_vector.is_(None),
        ),
    )
    processed_docs = 0
    while True:
        with Session(engine) as session, session.begin():
            documents = list(session.scalars(
                select(Document).where(needs_index).order_by(Document.id)
                .limit(32).with_for_update(skip_locked=True)
            ))
            if not documents:
                break
            doc_ids = [d.id for d in documents]
            session.execute(delete(DocumentChunk).where(DocumentChunk.document_id.in_(doc_ids)))
            session.add_all(index_documents(documents))
            upsert_vocab(session, vocab_for_documents(documents))
        processed_docs += len(documents)
        if processed_docs % 128 == 0:
            print(f"Indexed {processed_docs} documents", flush=True)

    needs_client_embed = Client.description.is_not(None) & (func.btrim(Client.description) != "") & Client.description_embedding.is_(None)
    processed_clients = 0
    while True:
        with Session(engine) as session, session.begin():
            clients = list(session.scalars(
                select(Client).where(needs_client_embed).order_by(Client.id)
                .limit(64).with_for_update(skip_locked=True)
            ))
            if not clients:
                break
            embed_client_descriptions(clients)
            upsert_vocab(session, vocab_for_clients(clients))
        processed_clients += len(clients)

    with Session(engine) as session:
        remaining_docs = session.scalar(select(func.count()).select_from(Document).where(needs_index))
        remaining_clients = session.scalar(select(func.count()).select_from(Client).where(needs_client_embed))
    print(
        f"Indexed {processed_docs} documents (remaining: {remaining_docs}) "
        f"and {processed_clients} client descriptions (remaining: {remaining_clients})",
        flush=True,
    )


if __name__ == "__main__":
    main()
