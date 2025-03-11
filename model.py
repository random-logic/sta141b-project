import os
from typing import Sequence
import ollama

from dotenv import load_dotenv
load_dotenv()

EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf')
LANGUAGE_MODEL = os.environ.get('LANGUAGE_MODEL', 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF')

class Model:
  def __init__(self, embedding_model = EMBEDDING_MODEL, language_model = LANGUAGE_MODEL):
    # Each element in the VECTOR_DB will be a tuple (chunk, embedding)
    # The embedding is a list of floats, for example: [0.1, 0.04, -0.34, 0.21, ...]
    self.vector_db = []
    self.embedding_model = embedding_model
    self.language_model = language_model

  def add_chunk_to_database(self, chunk: str | Sequence[str]):
    embedding = self.embed(chunk)
    self.vector_db.append((chunk, embedding))

  @staticmethod
  def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

  def retrieve(self, query: str, top_n = 1):
    query_embedding = self.embed(query)

    # temporary list to store (chunk, similarity) pairs
    similarities = []

    for chunk, embedding in self.vector_db:
      similarity = self.cosine_similarity(query_embedding, embedding)
      similarities.append((chunk, similarity))

    # sort by similarity in descending order, because higher similarity means more relevant chunks
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_n]

  def embed(self, text: str):
    embeddings = ollama.embed(model=self.embedding_model, input=text)['embeddings']
    return embeddings[0]