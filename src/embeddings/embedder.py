import openai
import numpy as np
from src.config import EMBED_MODEL

def generate_answer_openai(prompt, model="gpt-3.5-turbo", max_tokens=500):
    """Generate an answer using OpenAI's chat completion API."""
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "És um assistente de ensino especializado em regulamentos de construção."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

def get_embedding(text: str) -> list:
    """Get embedding vector from OpenAI model."""
    #print("Embedding input:", text, type(text))

    response = openai.embeddings.create(
        input=text,
        model=EMBED_MODEL
    )
    return response.data[0].embedding

def batch_embed(chunks: list) -> list:
    """Embed multiple chunks."""
    return [get_embedding(chunk) for chunk in chunks]
