"""
Simple, dependency-light text chunker.

Splits text into overlapping chunks of roughly `chunk_size` characters,
trying to break on paragraph/sentence boundaries rather than mid-word
where possible. Good enough for a demo RAG pipeline — production
systems often use token-aware chunkers (tiktoken) or semantic
chunking, but character-based chunking is easy to reason about and
works fine for most text.
"""


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try to end on a paragraph or sentence boundary instead of
        # cutting mid-sentence, by looking backward from `end`.
        if end < text_len:
            boundary = text.rfind("\n\n", start, end)
            if boundary == -1:
                boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward, leaving `overlap` characters of context so we
        # don't lose meaning that straddles a chunk boundary.
        start = end - overlap if end - overlap > start else end

    return chunks
