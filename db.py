"""
Postgres + pgvector connection and schema.

Table layout:
  document_chunks(id, source, chunk_index, content, embedding)

embedding is a 1536-dim vector because we use OpenAI's
text-embedding-3-small model, which outputs 1536 dimensions.
"""
import os
import psycopg2

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://docchat:docchat@localhost:5432/docchat"
)

EMBEDDING_DIM = 1536


def get_connection():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    return conn


def to_pgvector(vector: list[float]) -> str:
    """Convert a Python list of floats into pgvector's text literal
    format, e.g. [0.1, 0.2, 0.3] -> '[0.1,0.2,0.3]'.

    We pass vectors as strings (cast with ::vector in SQL) instead of
    relying on psycopg2's register_vector() adapter, which can
    silently fail to register depending on the pgvector-python/numpy
    versions installed — this approach has no such dependency.
    """
    return "[" + ",".join(str(x) for x in vector) + "]"


def init_schema():
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                source TEXT NOT NULL,
                chunk_index INT NOT NULL,
                content TEXT NOT NULL,
                embedding VECTOR({EMBEDDING_DIM})
            );
        """)
        # Index for fast approximate nearest-neighbor search.
        # ivfflat needs data in the table before it's useful, but
        # creating it up front is fine for a demo-scale dataset.
        cur.execute("""
            CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
            ON document_chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
    conn.close()
    print("Schema ready.")


def clear_documents():
    """Wipe all chunks — useful when re-running ingest.py from scratch."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("TRUNCATE document_chunks RESTART IDENTITY;")
    conn.close()


if __name__ == "__main__":
    init_schema()