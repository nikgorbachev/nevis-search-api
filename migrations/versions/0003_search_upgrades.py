"""Add fuzzystrmatch, normalized search vectors, vocab table, client description embeddings, and GIN/HNSW indexes."""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS fuzzystrmatch")

    op.add_column("clients", sa.Column("description_embedding", Vector(384), nullable=True))

    op.add_column("document_chunks", sa.Column("normalized_title", sa.Text(), nullable=False, server_default=""))
    op.add_column("document_chunks", sa.Column("normalized_passage", sa.Text(), nullable=False, server_default=""))
    op.add_column("document_chunks", sa.Column("search_vector", postgresql.TSVECTOR(), nullable=True))

    op.create_table(
        "search_vocab",
        sa.Column("word", sa.Text(), primary_key=True),
        sa.Column("doc_freq", sa.Integer(), nullable=False, server_default="1"),
    )

    op.execute("CREATE INDEX ix_document_chunks_search_vector ON document_chunks USING gin (search_vector)")
    op.execute("CREATE INDEX ix_document_chunks_norm_title_trgm ON document_chunks USING gin (normalized_title gin_trgm_ops)")
    op.execute("CREATE INDEX ix_document_chunks_norm_passage_trgm ON document_chunks USING gin (normalized_passage gin_trgm_ops)")
    op.execute("CREATE INDEX ix_document_chunks_embedding_hnsw ON document_chunks USING hnsw (embedding vector_cosine_ops)")
    op.execute("CREATE INDEX ix_clients_email_trgm ON clients USING gin (lower(email) gin_trgm_ops)")
    op.execute("CREATE INDEX ix_clients_name_trgm ON clients USING gin (lower(first_name || ' ' || last_name) gin_trgm_ops)")
    op.execute("CREATE INDEX ix_clients_desc_trgm ON clients USING gin (lower(coalesce(description, '')) gin_trgm_ops)")
    op.execute("CREATE INDEX ix_search_vocab_word_trgm ON search_vocab USING gin (word gin_trgm_ops)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_search_vocab_word_trgm")
    op.execute("DROP INDEX IF EXISTS ix_clients_desc_trgm")
    op.execute("DROP INDEX IF EXISTS ix_clients_name_trgm")
    op.execute("DROP INDEX IF EXISTS ix_clients_email_trgm")
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_norm_passage_trgm")
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_norm_title_trgm")
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_search_vector")
    op.drop_table("search_vocab")
    op.drop_column("document_chunks", "search_vector")
    op.drop_column("document_chunks", "normalized_passage")
    op.drop_column("document_chunks", "normalized_title")
    op.drop_column("clients", "description_embedding")
