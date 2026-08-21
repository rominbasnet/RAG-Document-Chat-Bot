"""
Simple, dependency-light text chunker.


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
