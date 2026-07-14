"""Build the NADRA knowledge base: PDFs -> clean English text -> chunks -> ChromaDB.

Extraction uses PyMuPDF. The source PDFs are bilingual, but the Urdu text is a
translation of the English on the same page (and in most files its font map is
broken beyond recovery), so ingestion keeps Latin-script text only.

Scanned form PDFs have no text layer; run transcribe_forms.py once first to
generate text sidecars in pdfs_data/transcribed/, which are ingested here too.

Run once (and re-run whenever the PDFs change):
    python src/ingest.py
"""

import re
import shutil
import unicodedata
from collections import Counter

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import kb

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 60  # drop near-empty chunks (page numbers, stray labels)

# Map each PDF to the NADRA service(s) it covers, used as retrieval metadata.
SERVICE_MAP = {
    "cnic-modification": "CNIC",
    "cnic-modification-nonprintable-17-11-2025": "CNIC",
    "conversion-to-smartid": "CNIC",
    "new-nicop": "NICOP",
    "nicop-complete-form-with-instruction": "NICOP",
    "new-smart-poc": "POC",
    "poc-complete-form-with-instruction": "POC",
    "new-crc-version-3-0": "CRC",
    "birth-registration": "CRC",
    "frc-guide-v2": "FRC",
    "renewal-guidelines": "General",
    "reprint-guide": "General",
    "payment-v4": "General",
    "application-tracking": "General",
    "appointment-scheduling": "General",
    "upload-document-guide": "General",
    "photo-guidelines": "General",
    "photgraph-guidelines-v4": "General",
    "fingerprint-guidelines": "General",
    "fingerprint-guidelines-v11": "General",
    "proof-of-life-certificate": "General",
    "registration-policy-6-0-1-english": "General",
    "death-registration": "General",
    "id-cancellation-death": "General",
    "id-cancellation-surrender": "General",
    "NADRA_Office_Locations_Pakistan": "General",
}

# Keep printable ASCII plus common typography; everything else (Urdu script,
# broken-font glyphs, symbols) becomes a space.
_NON_LATIN = re.compile(r"[^\x20-\x7E\n‘’“”–—•·…]+")


def clean_lines(text: str) -> list[str]:
    """Strip non-Latin noise and return the lines that still carry content."""
    text = unicodedata.normalize("NFKC", text)
    text = _NON_LATIN.sub(" ", text)
    lines = []
    for line in text.splitlines():
        # Table-of-contents entries ("Fees ............ 12") are pure noise.
        if re.search(r"\.{4,}", line):
            continue
        # Broken Urdu ligatures surface as stray "nj" tokens and empty parens.
        line = re.sub(r"\bnj\b|\(\s*\)", " ", line)
        line = re.sub(r"\s+", " ", line).strip()
        digits = sum(c.isdigit() for c in line)
        # A real content line has an actual word (3+ consecutive letters) or
        # numeric data (fees, dates, phone). Broken Urdu glyphs that map into
        # Latin ("nj ( ) nj") never form real words, so this drops them.
        if re.search(r"[A-Za-z]{3,}", line) or digits >= 2:
            lines.append(line)
    return lines


def _boiler_key(line: str) -> str:
    # Ignore digits so "User Guideline 7" and "User Guideline 13" match.
    return re.sub(r"\s+", " ", re.sub(r"\d+", "", line)).strip()


def remove_boilerplate(pages_lines: list[list[str]]) -> list[list[str]]:
    """Drop short lines repeated on most pages (running headers/footers)."""
    n_pages = len(pages_lines)
    if n_pages < 4:
        return pages_lines
    counts = Counter(_boiler_key(line) for lines in pages_lines for line in set(lines))
    threshold = max(3, int(0.6 * n_pages))
    boiler = {k for k, c in counts.items() if c >= threshold and len(k) <= 60}
    return [
        [l for l in lines if _boiler_key(l) not in boiler]
        for lines in pages_lines
    ]


def load_pdf_documents() -> list[Document]:
    docs = []
    for pdf_path in sorted(kb.PDF_DIR.glob("*.pdf")):
        service = SERVICE_MAP.get(pdf_path.stem, "General")
        with fitz.open(pdf_path) as pdf:
            pages_lines = [clean_lines(page.get_text()) for page in pdf]
        pages_lines = remove_boilerplate(pages_lines)
        kept = 0
        for page_num, lines in enumerate(pages_lines, start=1):
            content = "\n".join(lines)
            if len(content) < MIN_CHUNK_CHARS:
                continue
            docs.append(Document(
                page_content=content,
                metadata={"source": pdf_path.name, "service": service, "page": page_num},
            ))
            kept += 1
        note = "" if kept else "  <-- no text (scanned? see transcribe_forms.py)"
        print(f"  {pdf_path.name}: {kept}/{len(pages_lines)} pages kept{note}")
    return docs


def load_transcribed_documents() -> list[Document]:
    """Load text sidecars produced by transcribe_forms.py ([Page N] separated)."""
    docs = []
    if not kb.TRANSCRIBED_DIR.exists():
        return docs
    for txt_path in sorted(kb.TRANSCRIBED_DIR.glob("*.txt")):
        service = SERVICE_MAP.get(txt_path.stem, "General")
        text = txt_path.read_text(encoding="utf-8")
        for match in re.finditer(r"\[Page (\d+)\]\n(.*?)(?=\n\[Page \d+\]|\Z)", text, re.S):
            page_num, content = int(match.group(1)), match.group(2).strip()
            if len(content) < MIN_CHUNK_CHARS:
                continue
            docs.append(Document(
                page_content=content,
                metadata={"source": f"{txt_path.stem}.pdf", "service": service, "page": page_num},
            ))
        print(f"  {txt_path.name}: transcription loaded")
    return docs


def main():
    print(f"Loading PDFs from {kb.PDF_DIR} ...")
    docs = load_pdf_documents() + load_transcribed_documents()
    print(f"\nLoaded {len(docs)} pages total.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_CHARS]
    print(f"Split into {len(chunks)} chunks (>= {MIN_CHUNK_CHARS} chars).")

    if kb.VECTOR_DIR.exists():
        print("Removing old vector store ...")
        shutil.rmtree(kb.VECTOR_DIR)

    print(f"Embedding with {kb.EMBEDDING_MODEL} via ONNX (local CPU) ...")
    kb.build_vectorstore(chunks)
    print(f"\nDone. Vector store persisted to {kb.VECTOR_DIR}")


if __name__ == "__main__":
    main()
