"""
Wraps OpenAI's embeddings endpoint. Kept in its own module so both
ingest.py (embedding documents) and main.py (embedding the user's
question) use the exact same model/settings — mismatched embedding
models between ingestion and query time is a common RAG bug.
"""
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dimensions, cheap, fast


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of strings. OpenAI allows batching, which is much
    faster and cheaper than one API call per chunk.
    """
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]
