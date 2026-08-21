"""
Ingestion pipeline: reads files from ./docs, chunks them, embeds each
chunk, and stores them in Postgres/pgvector.

Run this whenever you add or change documents:
    python ingest.py

Supports .pdf, .txt, and .md files. To add Confluence/Notion later,
you'd write a small function that pulls pages via their API and
returns the same (source, text) pairs — chunking/embedding/storage
stays identical.
"""
import os
import sys
from pypdf import PdfReader

from chunking import chunk_text
from embeddings import embed_texts
from db import get_connection, init_schema, clear_documents, to_pgvector

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
BATCH_SIZE = 50  # chunks per embedding API call


def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_documents() -> list[tuple[str, str]]:
    """Returns a list of (source_name, full_text) pairs from ./docs."""
    documents = []
    if not os.path.isdir(DOCS_DIR):
        print(f"No docs directory found at {DOCS_DIR}")
        return documents

    for filename in sorted(os.listdir(DOCS_DIR)):
        path = os.path.join(DOCS_DIR, filename)
        if not os.path.isfile(path):
            continue

        ext = filename.lower().rsplit(".", 1)[-1]
        try:
            if ext == "pdf":
                text = load_pdf(path)
            elif ext in ("txt", "md"):
                text = load_text(path)
            else:
                continue  # skip unsupported file types
        except Exception as e:
            print(f"  Skipping {filename}: failed to read ({e})")
            continue

        if text.strip():
            documents.append((filename, text))
        else:
            print(f"  Skipping {filename}: no extractable text")

    return documents


def ingest():
    init_schema()

    documents = load_documents()
    if not documents:
        print(f"No documents found in {DOCS_DIR}. Add .pdf/.txt/.md files and re-run.")
        return

    print(f"Found {len(documents)} document(s). Clearing old chunks...")
    clear_documents()

    conn = get_connection()
    total_chunks = 0

    for source, text in documents:
        chunks = chunk_text(text)
        print(f"  {source}: {len(chunks)} chunks")

        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[batch_start: batch_start + BATCH_SIZE]
            vectors = embed_texts(batch)

            with conn.cursor() as cur:
                for i, (chunk, vector) in enumerate(zip(batch, vectors)):
                    cur.execute(
                        """
                        INSERT INTO document_chunks (source, chunk_index, content, embedding)
                        VALUES (%s, %s, %s, %s::vector)
                        """,
                        (source, batch_start + i, chunk, to_pgvector(vector)),
                    )
            total_chunks += len(batch)

    conn.close()
    print(f"Done. Ingested {total_chunks} chunks from {len(documents)} document(s).")


if __name__ == "__main__":
    ingest()