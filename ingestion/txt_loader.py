from datetime import datetime
from pathlib import Path

from llama_index.core import Document


def load_txt(file_path: str) -> list[Document]:
    file_name = Path(file_path).name
    ingested_at = datetime.now().isoformat()
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    return [Document(
        text=text,
        metadata={"file_name": file_name, "page": None, "content_type": "text", "ingested_at": ingested_at},
    )]
