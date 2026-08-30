import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import QDRANT_URL, COLLECTION_NAME, EMBED_MODEL_NAME
from ingestion.pdf_loader import load_pdf
from ingestion.md_loader import load_md
from ingestion.txt_loader import load_txt
from ingestion.naming import make_ingest_name

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ingest")

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".md"}


def _ingest(file_path: str, client: QdrantClient, embed_model: HuggingFaceEmbedding) -> None:
    src = Path(file_path)
    unique_name = make_ingest_name(src.name)

    with tempfile.TemporaryDirectory() as tmp_dir:
        staged_path = Path(tmp_dir) / unique_name
        shutil.copyfile(src, staged_path)
        file_path = str(staged_path)

        if file_path.endswith(".pdf"):
            documents = load_pdf(file_path)
        elif file_path.endswith(".md"):
            documents = load_md(file_path)
        else:
            documents = load_txt(file_path)

        table_docs = [d for d in documents if d.metadata.get("content_type") == "table"]
        other_docs = [d for d in documents if d.metadata.get("content_type") != "table"]

        splitter = SentenceSplitter()

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

    print(f"Ingested '{src.name}' as '{unique_name}' into Qdrant collection '{COLLECTION_NAME}'.")


def ingest(file_path: str) -> None:
    client = QdrantClient(url=QDRANT_URL)
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    _ingest(file_path, client, embed_model)


def wipe_collection(client: QdrantClient, collection_name: str) -> bool:
    if not client.collection_exists(collection_name):
        logger.warning(f"Wipe skipped: collection '{collection_name}' does not exist.")
        return False
    client.delete_collection(collection_name)
    logger.info(f"Wipe succeeded: collection '{collection_name}' deleted.")
    return True


def resolve_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS)
    if not files:
        logger.warning(f"No files with extensions {sorted(ALLOWED_EXTENSIONS)} found under '{path}'.")
        return []

    print(f"Found {len(files)} file(s) under '{path}':")
    for f in files:
        print(f"  {f.name}")
    confirm = input("Ingest all of these files? [y/N] ").strip().lower()
    if confirm != "y":
        logger.info("Ingestion cancelled by user.")
        return []
    return files


def run_ingestion(path_arg: str, client: QdrantClient, embed_model: HuggingFaceEmbedding) -> None:
    path = Path(path_arg)
    if not path.exists():
        logger.error(f"Ingestion failed: path '{path}' does not exist.")
        return

    files = resolve_files(path)
    if not files:
        return

    for f in files:
        try:
            _ingest(str(f), client, embed_model)
            logger.info(f"Ingestion succeeded: '{f.name}'.")
        except Exception:
            logger.exception(f"Ingestion failed: '{f.name}'.")


def restart_api() -> None:
    logger.info("Restarting 'api' container...")
    try:
        subprocess.run(["docker", "compose", "restart", "api"], check=True)
        logger.info("Restart succeeded: 'api' container restarted.")
    except subprocess.CalledProcessError:
        logger.exception("Restart failed: 'docker compose restart api' returned a non-zero exit code.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest file(s), or wipe/reingest/restart the stack.")
    parser.add_argument("file_path", nargs="?", help="Path to a single file to ingest (original single-file usage).")
    wipe_group = parser.add_mutually_exclusive_group()
    wipe_group.add_argument("-c", "--collection", help="Name of the Qdrant collection to wipe.")
    wipe_group.add_argument("--all", action="store_true", help="Wipe all Qdrant collections.")
    parser.add_argument("-p", "--path", help="File or directory path to ingest.")
    args = parser.parse_args()

    if args.file_path and not (args.collection or args.all or args.path):
        # Original single-file usage: `python ingest.py file_path`
        ingest(args.file_path)
        sys.exit(0)

    qdrant_client = QdrantClient(url=QDRANT_URL)
    wiped = False

    if args.collection:
        print(f"About to wipe collection: {args.collection}")
        if input("Proceed? [y/N] ").strip().lower() == "y":
            wiped = wipe_collection(qdrant_client, args.collection)
        else:
            logger.info("Wipe cancelled by user.")
    elif args.all:
        collections = [c.name for c in qdrant_client.get_collections().collections]
        if not collections:
            logger.warning("Wipe skipped: no collections exist.")
        else:
            print("About to wipe ALL collections:")
            for name in collections:
                print(f"  {name}")
            if input("Proceed? [y/N] ").strip().lower() == "y":
                for name in collections:
                    wiped = wipe_collection(qdrant_client, name) or wiped
            else:
                logger.info("Wipe cancelled by user.")
    else:
        logger.info("Wipe skipped: no -c/--collection or --all provided.")

    if args.path:
        embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
        run_ingestion(args.path, qdrant_client, embed_model)
    else:
        logger.info("Ingestion skipped: no -p/--path provided.")

    if wiped:
        restart_api()
    else:
        logger.info("Restart skipped: no wipe was performed.")
