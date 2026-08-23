import pdfplumber
from pathlib import Path
from llama_index.core import Document


def _table_to_markdown(table: list[list[str]]) -> str:
    rows = [[cell or "" for cell in row] for row in table]
    header, *body = rows
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def load_pdf(file_path: str) -> list[Document]:
    file_name = Path(file_path).name
    documents = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(Document(
                    text=text,
                    metadata={"file_name": file_name, "page": page_num, "content_type": "text"},
                ))

            for table in page.extract_tables():
                if len(table) < 2:
                    continue
                md_table = _table_to_markdown(table)
                documents.append(Document(
                    text=md_table,
                    metadata={"file_name": file_name, "page": page_num, "content_type": "table"},
                ))

    return documents
