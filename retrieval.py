"""
Retrieval: given a query embedding, find the most similar chunks
using pgvector's cosine distance operator (<=>).
"""
from db import get_connection, to_pgvector


def retrieve_chunks(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    conn = get_connection()
    vector_literal = to_pgvector(query_embedding)

    with conn.cursor() as cur:
        # <=> is pgvector's cosine distance operator: smaller = more similar.
        # Cast explicitly with ::vector so Postgres knows how to compare
        # it against the embedding column (see db.to_pgvector for why).
        cur.execute(
            """
            SELECT source, chunk_index, content, embedding <=> %s::vector AS distance
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (vector_literal, vector_literal, top_k),
        )
        rows = cur.fetchall()
    conn.close()

    return [
        {
            "source": r[0],
            "chunk_index": r[1],
            "content": r[2],
            "distance": float(r[3]),
        }
        for r in rows
    ]