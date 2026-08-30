import re
from datetime import datetime
from pathlib import Path

from llama_index.core import Document

_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")


def load_md(file_path: str) -> list[Document]:
    file_name = Path(file_path).name
    ingested_at = datetime.now().isoformat()
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    documents = []
    buffer = []
    table_buffer = []

    def flush_text():
        if buffer:
            text = "".join(buffer).strip()
            if text:
                documents.append(Document(
                    text=text,
                    metadata={"file_name": file_name, "page": None, "content_type": "text", "ingested_at": ingested_at},
                ))
            buffer.clear()

    def flush_table():
        if table_buffer:
            text = "".join(table_buffer).strip()
            documents.append(Document(
                text=text,
                metadata={"file_name": file_name, "page": None, "content_type": "table", "ingested_at": ingested_at},
            ))
            table_buffer.clear()

    for line in lines:
        if _TABLE_LINE.match(line):
            flush_text()
            table_buffer.append(line)
        else:
            flush_table()
            buffer.append(line)

    flush_table()
    flush_text()

    return documents
