import sys
from pathlib import Path

import dagster as dg
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import COLLECTION_NAME, EMBED_MODEL_NAME, QDRANT_URL
from ingestion.pdf_loader import load_pdf
from ingestion.md_loader import load_md
from ingestion.txt_loader import load_txt


class IngestConfig(dg.Config):
    file_path: str


@dg.asset
def raw_documents(config: IngestConfig) -> list[Document]:
    if config.file_path.endswith(".pdf"):
        return load_pdf(config.file_path)
    if config.file_path.endswith(".md"):
        return load_md(config.file_path)
    return load_txt(config.file_path)


@dg.asset
def chunks(raw_documents: list[Document]) -> list[BaseNode]:
    table_docs = [d for d in raw_documents if d.metadata.get("content_type") == "table"]
    other_docs = [d for d in raw_documents if d.metadata.get("content_type") != "table"]

    splitter = SentenceSplitter()
    nodes = splitter.get_nodes_from_documents(other_docs)
    nodes += [TextNode(text=d.text, metadata=d.metadata) for d in table_docs]
    return nodes


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
