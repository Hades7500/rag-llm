import hashlib
import io


def parse_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    if name.endswith(".pdf"):
        return _parse_pdf(data)
    if name.endswith(".docx"):
        return _parse_docx(data)
    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="replace")

    raise ValueError(
        f"Unsupported file type: '{uploaded_file.name}'. Supported: PDF, DOCX, TXT, MD."
    )


def _parse_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "pypdf is required for PDF support: pip install pypdf"
        ) from exc

    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _parse_docx(data: bytes) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise ImportError(
            "python-docx is required for DOCX support: pip install python-docx"
        ) from exc

    doc = Document(io.BytesIO(data))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 100) -> list[str]:
    """Split text into overlapping fixed-size chunks."""
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    step = max(chunk_size - overlap, 1)

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step

    return chunks


def make_ids(filename: str, raw_bytes: bytes, n: int) -> list[str]:
    content_hash = hashlib.md5(raw_bytes).hexdigest()[:12]
    return [f"{content_hash}_{i}" for i in range(n)]
