import sys
from pathlib import Path

import dagster as dg
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import COLLECTION_NAME, EMBED_MODEL_NAME, QDRANT_URL


class IngestConfig(dg.Config):
    file_path: str


@dg.asset
def raw_documents(config: IngestConfig) -> list[Document]:
    return SimpleDirectoryReader(input_files=[config.file_path]).load_data()


@dg.asset
def chunks(raw_documents: list[Document]) -> list[BaseNode]:
    splitter = SentenceSplitter()
    return splitter.get_nodes_from_documents(raw_documents)


@dg.asset
def embeddings(chunks: list[BaseNode]) -> list[BaseNode]:
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    texts = [node.get_content() for node in chunks]
    vectors = embed_model.get_text_embedding_batch(texts)
    for node, vector in zip(chunks, vectors):
        node.embedding = vector
    return chunks


@dg.asset
def qdrant_index(embeddings: list[BaseNode]) -> None:
    client = QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    vector_store.add(embeddings)


ingest_job = dg.define_asset_job(
    name="ingest_job", selection=[raw_documents, chunks, embeddings, qdrant_index]
)

defs = dg.Definitions(
    assets=[raw_documents, chunks, embeddings, qdrant_index], jobs=[ingest_job]
)
