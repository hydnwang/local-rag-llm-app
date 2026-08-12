import argparse

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "documents"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def ingest(file_path: str) -> None:
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

    splitter = SentenceSplitter()
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

    client = QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
    )

    print(f"Ingested '{file_path}' into Qdrant collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a text file into Qdrant.")
    parser.add_argument("file_path", help="Path to the .txt file to ingest")
    args = parser.parse_args()

    ingest(args.file_path)
