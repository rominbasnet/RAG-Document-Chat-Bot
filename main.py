"""
FastAPI app: ask a question, get an answer grounded in your documents.

No frontend needed — use the auto-generated Swagger UI at /docs to
test it, same as the support-copilot project.
"""
import os
from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI

from embeddings import embed_text
from retrieval import retrieve_chunks
load_dotenv()
app = FastAPI(title="Chat With Our Docs")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

CHAT_MODEL = "gpt-4o"

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using
ONLY the provided context excerpts from internal documents.

Rules:
- If the answer isn't in the context, say you don't have that
  information rather than guessing.
- Keep answers concise and directly address the question.
- After your answer, do not fabricate source names — the API response
  already includes real sources separately.
"""


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5


class Source(BaseModel):
    source: str
    chunk_index: int
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # 1. Embed the question with the SAME model used at ingest time
    query_embedding = embed_text(req.question)

    # 2. Retrieve the most relevant chunks from pgvector
    chunks = retrieve_chunks(query_embedding, top_k=req.top_k)

    if not chunks:
        return ChatResponse(
            answer="I don't have any documents ingested yet — run "
                   "ingest.py first.",
            sources=[],
        )

    # 3. Build context block for the LLM
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}, chunk {c['chunk_index']}]\n{c['content']}"
        for c in chunks
    )

    # 4. Ask the LLM to answer using only that context
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.question}"},
        ],
    )
    answer = response.choices[0].message.content

    sources = [
        Source(source=c["source"], chunk_index=c["chunk_index"],
               excerpt=c["content"][:200] + ("..." if len(c["content"]) > 200 else ""))
        for c in chunks
    ]

    return ChatResponse(answer=answer, sources=sources)


@app.get("/health")
def health():
    return {"status": "ok"}
