import tiktoken

def chunk_text(text: str, max_tokens: int = 500, overlap: int = 50, model="gpt-3.5-turbo") -> list:
    """Split text into overlapping chunks based on tokens."""
    enc = tiktoken.encoding_for_model(model)
    print(text)
    print(type(text))
    tokens = enc.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        start += max_tokens - overlap

    return chunks
