import os

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "documents"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = "llama3.1:8b"
TOP_K = 8

DAGSTER_HOST = os.environ.get("DAGSTER_HOST", "localhost")
DAGSTER_PORT = 3000
