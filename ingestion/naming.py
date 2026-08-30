import uuid


def make_ingest_name(filename: str) -> str:
    return f"{uuid.uuid4().hex}_{filename}"
