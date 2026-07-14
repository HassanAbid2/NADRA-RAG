"""One-time transcription of scanned (image-only) form PDFs via Groq vision.

Renders each page to PNG and asks a vision model to transcribe the English
text. Output goes to pdfs_data/transcribed/<stem>.txt with [Page N] markers,
which ingest.py picks up automatically. Skips PDFs already transcribed.

Usage:
    python src/transcribe_forms.py
"""

import base64
from pathlib import Path

import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

import kb

load_dotenv(kb.PROJECT_ROOT / ".env")

SCANNED_PDFS = [
    "nicop-complete-form-with-instruction.pdf",
    "poc-complete-form-with-instruction.pdf",
]
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

PROMPT = """This image is a page from an official NADRA (National Database and \
Registration Authority, Pakistan) application form with instructions. Transcribe \
ALL the English text on the page: form field names, section headings, \
instructions, notes, fee amounts, document requirements — preserving the reading \
order and structure (use plain lines and simple lists). The page also contains \
Urdu text which is a translation of the English; skip the Urdu. Do not describe \
the page or add commentary — output only the transcribed text."""


def transcribe_page(llm: ChatGroq, pixmap_png: bytes) -> str:
    b64 = base64.b64encode(pixmap_png).decode()
    message = HumanMessage(content=[
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ])
    return llm.invoke([message]).content.strip()


def main():
    kb.TRANSCRIBED_DIR.mkdir(parents=True, exist_ok=True)
    llm = ChatGroq(model=VISION_MODEL, temperature=0)

    for pdf_name in SCANNED_PDFS:
        out_path = kb.TRANSCRIBED_DIR / f"{Path(pdf_name).stem}.txt"
        if out_path.exists():
            print(f"{pdf_name}: already transcribed, skipping.")
            continue
        pdf_path = kb.PDF_DIR / pdf_name
        if not pdf_path.exists():
            print(f"{pdf_name}: PDF not found, skipping.")
            continue

        print(f"{pdf_name}: transcribing ...")
        sections = []
        with fitz.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                png = page.get_pixmap(dpi=150).tobytes("png")
                text = transcribe_page(llm, png)
                sections.append(f"[Page {page_num}]\n{text}")
                print(f"  page {page_num}: {len(text)} chars")
        out_path.write_text("\n\n".join(sections), encoding="utf-8")
        print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
