import argparse
import sys
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import QDRANT_URL, COLLECTION_NAME, EMBED_MODEL_NAME
from ingestion.pdf_loader import load_pdf
from ingestion.md_loader import load_md


def ingest(file_path: str) -> None:
    if file_path.endswith(".pdf"):
        documents = load_pdf(file_path)
    elif file_path.endswith(".md"):
        documents = load_md(file_path)
    else:
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

    table_docs = [d for d in documents if d.metadata.get("content_type") == "table"]
    other_docs = [d for d in documents if d.metadata.get("content_type") != "table"]

    splitter = SentenceSplitter()
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

    client = QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        other_docs,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
    )

    if table_docs:
        table_nodes = [
            TextNode(text=d.text, metadata=d.metadata) for d in table_docs
        ]
        index.insert_nodes(table_nodes)

    print(f"Ingested '{file_path}' into Qdrant collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a text file into Qdrant.")
    parser.add_argument("file_path", help="Path to the .txt file to ingest")
    args = parser.parse_args()

    ingest(args.file_path)
