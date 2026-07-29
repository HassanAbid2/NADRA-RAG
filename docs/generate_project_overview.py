"""Generate the NADRA RAG project overview PDF.

The file uses PyMuPDF, which is already part of the project's dependency set.
Run from the repository root:
    .venv\\Scripts\\python docs\\generate_project_overview.py
"""

from pathlib import Path
import textwrap

import fitz


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs" / "NADRA_RAG_Project_Overview.pdf"

PAGE = fitz.paper_rect("a4")
W, H = PAGE.width, PAGE.height
MARGIN = 46

GREEN = (0.03, 0.25, 0.18)
GREEN_2 = (0.08, 0.43, 0.31)
MINT = (0.88, 0.96, 0.92)
PALE = (0.96, 0.98, 0.97)
GOLD = (0.92, 0.68, 0.18)
INK = (0.10, 0.15, 0.13)
MUTED = (0.34, 0.42, 0.38)
LINE = (0.80, 0.86, 0.83)
WHITE = (1, 1, 1)


def text_width(text, size, font="helv"):
    return fitz.get_text_length(text, fontname=font, fontsize=size)


def wrap(text, width, size, font="helv"):
    words = text.split()
    lines, current = [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if not current or text_width(trial, size, font) <= width:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_text(page, x, y, text, size=10, color=INK, font="helv"):
    page.insert_text((x, y), text, fontsize=size, fontname=font, color=color)


def draw_wrapped(page, x, y, text, width, size=10, leading=None, color=INK,
                 font="helv"):
    leading = leading or size * 1.42
    for line in wrap(text, width, size, font):
        draw_text(page, x, y, line, size, color, font)
        y += leading
    return y


def draw_bullets(page, x, y, items, width, size=9.5, gap=6):
    for item in items:
        draw_text(page, x, y, "-", size, GREEN_2, "hebo")
        y = draw_wrapped(page, x + 14, y, item, width - 14, size, size * 1.4)
        y += gap
    return y


def rounded_box(page, rect, fill=PALE, stroke=LINE, radius=8):
    page.draw_rect(rect, color=stroke, fill=fill, width=0.8,
                   radius=min(radius / 100, 0.5))


def section_label(page, x, y, text):
    draw_text(page, x, y, text.upper(), 8.3, GREEN_2, "hebo")


def page_header(page, number, section):
    page.draw_rect(fitz.Rect(0, 0, W, 8), color=GREEN, fill=GREEN)
    draw_text(page, MARGIN, 31, "NADRA GUIDE", 8.5, GREEN, "hebo")
    draw_text(page, W - MARGIN - text_width(section, 8.5), 31, section, 8.5, MUTED)
    page.draw_line((MARGIN, 41), (W - MARGIN, 41), color=LINE, width=0.7)
    footer = f"Project overview  |  July 2026  |  {number}"
    draw_text(page, W - MARGIN - text_width(footer, 8), H - 25, footer, 8, MUTED)


def heading(page, y, title, subtitle=None):
    draw_text(page, MARGIN, y, title, 25, GREEN, "hebo")
    y += 22
    if subtitle:
        y = draw_wrapped(page, MARGIN, y, subtitle, W - 2 * MARGIN, 10.5, 15, MUTED)
    return y + 12


def page_one(doc):
    page = doc.new_page(width=W, height=H)
    page.draw_rect(PAGE, color=GREEN, fill=GREEN)
    page.draw_circle((W - 60, 90), 130, color=GREEN_2, fill=GREEN_2)
    page.draw_circle((W - 25, 45), 70, color=GOLD, fill=GOLD)

    draw_text(page, MARGIN, 85, "PROJECT EXPLAINER", 10, MINT, "hebo")
    draw_text(page, MARGIN, 136, "NADRA Guide", 37, WHITE, "hebo")
    draw_text(page, MARGIN, 176, "RAG Assistant", 37, WHITE, "hebo")
    y = draw_wrapped(
        page, MARGIN, 215,
        "What the project does, the technology behind it, and how the repository is organized.",
        430, 14, 20, MINT,
    )

    card = fitz.Rect(MARGIN, y + 34, W - MARGIN, y + 190)
    page.draw_rect(card, color=WHITE, fill=WHITE, radius=0.08)
    draw_text(page, MARGIN + 25, card.y0 + 31, "ONE-SENTENCE SUMMARY", 8.5, GREEN_2, "hebo")
    draw_wrapped(
        page, MARGIN + 25, card.y0 + 65,
        "A multilingual web and command-line assistant that retrieves relevant passages "
        "from official NADRA documents and asks a Groq-hosted language model to produce "
        "clear, source-aware answers about identity services.",
        card.width - 50, 13, 19, INK, "hebo",
    )

    y2 = card.y1 + 42
    draw_text(page, MARGIN, y2, "COVERS", 8.5, MINT, "hebo")
    chips = ["CNIC", "NICOP", "POC", "CRC", "FRC"]
    x = MARGIN
    for chip in chips:
        cw = text_width(chip, 10, "hebo") + 28
        page.draw_rect(fitz.Rect(x, y2 + 13, x + cw, y2 + 43),
                       color=GREEN_2, fill=GREEN_2, radius=0.45)
        draw_text(page, x + 14, y2 + 33, chip, 10, WHITE, "hebo")
        x += cw + 9

    draw_text(page, MARGIN, H - 45, "Prepared from the repository's current source code and configuration.",
              8.5, MINT)


def page_two(doc):
    page = doc.new_page(width=W, height=H)
    page_header(page, 2, "What it does")
    y = heading(
        page, 78, "1. What does this project do?",
        "It turns a collection of NADRA reference documents into a conversational guidance system.",
    )

    section_label(page, MARGIN, y, "User-facing behavior")
    y += 22
    y = draw_bullets(page, MARGIN, y, [
        "Answers questions about CNIC, NICOP, POC, CRC and FRC: eligibility, required documents, fees, timelines, procedures and office locations.",
        "Understands English, Urdu script and Roman Urdu, then responds in the user's language style.",
        "Carries up to 12 recent chat messages so follow-up questions and translation requests can use conversation context.",
        "Shows the source document and page behind an answer when the generated response actually uses them.",
        "Refuses out-of-scope or unsupported requests instead of intentionally filling gaps with general knowledge.",
    ], W - 2 * MARGIN)

    y += 5
    section_label(page, MARGIN, y, "How one question is answered")
    y += 16

    labels = [
        ("1", "Browser / CLI", "Citizen asks a question"),
        ("2", "Language + query handling", "Classify language; normalize common terms"),
        ("3", "Hybrid retrieval", "Vector similarity + BM25 + intent boosts"),
        ("4", "Groq LLM", "Generate an answer only from retrieved context"),
        ("5", "Response", "Return answer plus deduplicated source pages"),
    ]
    box_w = 92
    gap = 9
    x = MARGIN
    for number, title, body in labels:
        rect = fitz.Rect(x, y, x + box_w, y + 117)
        rounded_box(page, rect, WHITE, LINE, 7)
        page.draw_circle((x + 16, y + 17), 10, color=GREEN, fill=GREEN)
        draw_text(page, x + 13.2, y + 20.5, number, 8, WHITE, "hebo")
        draw_wrapped(page, x + 10, y + 46, title, box_w - 20, 9.2, 12, GREEN, "hebo")
        draw_wrapped(page, x + 10, y + 77, body, box_w - 20, 7.7, 10.5, MUTED)
        if number != "5":
            draw_text(page, x + box_w + 1, y + 61, ">", 12, GOLD, "hebo")
        x += box_w + gap

    y += 142
    section_label(page, MARGIN, y, "Important boundary")
    y += 15
    rect = fitz.Rect(MARGIN, y, W - MARGIN, y + 86)
    rounded_box(page, rect, MINT, GREEN_2, 8)
    draw_wrapped(
        page, MARGIN + 18, y + 25,
        "This is an information assistant, not an official transaction system. It does not issue cards, access citizen records, verify identities or replace NADRA. Answer quality depends on the documents loaded into the local knowledge base and availability of the Groq API.",
        rect.width - 36, 9.7, 14, INK,
    )


def page_three(doc):
    page = doc.new_page(width=W, height=H)
    page_header(page, 3, "Technology")
    y = heading(
        page, 78, "2. What tech stack does it use?",
        "A Python RAG backend, a lightweight browser frontend and a locally persisted retrieval index.",
    )

    rows = [
        ("Backend API", "Python, FastAPI, Uvicorn, Pydantic", "Validates chat requests, exposes /api/chat and /api/health, and serves the frontend."),
        ("RAG orchestration", "LangChain Core, langchain-groq", "Builds prompts and calls Groq's Llama 3.3 70B Versatile model at temperature 0."),
        ("Semantic search", "FastEmbed, ONNX Runtime, multilingual-e5-small", "Creates 384-dimensional multilingual embeddings locally on CPU."),
        ("Keyword search", "rank-bm25", "Finds exact terms such as service acronyms, fee labels and form fields."),
        ("Vector storage", "NumPy + JSON files", "Stores embeddings.npy, documents.json and manifest.json under data/vector_store."),
        ("Document processing", "PyMuPDF, LangChain text splitters", "Extracts PDF pages, cleans text and splits it into 1,000-character chunks with 150-character overlap."),
        ("Frontend", "HTML, CSS, vanilla JavaScript", "Provides the responsive chat interface, local conversation state and source display."),
        ("Frontend tooling", "Vite 7 (development dependency)", "Runs a dev server and production build; Node.js is optional for the normal FastAPI run."),
        ("Configuration", "python-dotenv + .env", "Loads GROQ_API_KEY; secrets remain outside committed source."),
    ]

    col1, col2 = 112, 172
    table_x = MARGIN
    table_w = W - 2 * MARGIN
    row_heights = [58, 58, 60, 53, 57, 64, 57, 61, 53]
    page.draw_rect(fitz.Rect(table_x, y, table_x + table_w, y + 27),
                   color=GREEN, fill=GREEN, radius=0.08)
    draw_text(page, table_x + 9, y + 18, "LAYER", 8, WHITE, "hebo")
    draw_text(page, table_x + col1 + 9, y + 18, "TECHNOLOGY", 8, WHITE, "hebo")
    draw_text(page, table_x + col1 + col2 + 9, y + 18, "ROLE", 8, WHITE, "hebo")
    y += 27
    for index, ((layer, tech, role), rh) in enumerate(zip(rows, row_heights)):
        fill = WHITE if index % 2 == 0 else PALE
        page.draw_rect(fitz.Rect(table_x, y, table_x + table_w, y + rh),
                       color=LINE, fill=fill, width=0.6)
        page.draw_line((table_x + col1, y), (table_x + col1, y + rh), color=LINE, width=0.5)
        page.draw_line((table_x + col1 + col2, y), (table_x + col1 + col2, y + rh), color=LINE, width=0.5)
        draw_wrapped(page, table_x + 9, y + 17, layer, col1 - 18, 8.6, 11, GREEN, "hebo")
        draw_wrapped(page, table_x + col1 + 9, y + 17, tech, col2 - 18, 8.1, 10.5, INK)
        draw_wrapped(page, table_x + col1 + col2 + 9, y + 17, role,
                     table_w - col1 - col2 - 18, 7.9, 10.5, MUTED)
        y += rh

    y += 17
    draw_text(page, MARGIN, y, "Deployment profile:", 9, GREEN, "hebo")
    draw_wrapped(
        page, MARGIN + 89, y,
        "single FastAPI process for the standard setup; embeddings and the vector index are local, while answer generation is remote through Groq.",
        W - 2 * MARGIN - 89, 9, 12.5, MUTED,
    )


def page_four(doc):
    page = doc.new_page(width=W, height=H)
    page_header(page, 4, "Structure")
    y = heading(
        page, 78, "3. How is the project structured?",
        "The repository separates web delivery, RAG logic, knowledge-base artifacts, source documents and evaluation utilities.",
    )

    tree = [
        ("NADRA-RAG-main/", GREEN, "hebo"),
        ("|-- app.py                 FastAPI entry point and static-file server", INK, "cour"),
        ("|-- src/", GREEN_2, "cobo"),
        ("|   |-- rag_pipeline.py     language policy, prompting, retrieval and answer flow", INK, "cour"),
        ("|   |-- kb.py               embeddings, local vector store and hybrid retriever", INK, "cour"),
        ("|   |-- ingest.py           PDF/text loading, cleaning, chunking and indexing", INK, "cour"),
        ("|   |-- chat_cli.py         terminal chat interface", INK, "cour"),
        ("|   `-- transcribe_forms.py scanned-form transcription helper", INK, "cour"),
        ("|-- frontend/", GREEN_2, "cobo"),
        ("|   |-- index.html          interface markup", INK, "cour"),
        ("|   |-- styles.css          responsive presentation", INK, "cour"),
        ("|   |-- main.js             chat state, API calls and source rendering", INK, "cour"),
        ("|   `-- package.json        optional Vite development/build setup", INK, "cour"),
        ("|-- pdfs_data/              source PDFs and transcribed text sidecars", INK, "cour"),
        ("|-- data/", GREEN_2, "cobo"),
        ("|   |-- vector_store/       active documents, vectors and model manifest", INK, "cour"),
        ("|   `-- fastembed_cache/    cached ONNX embedding model", INK, "cour"),
        ("|-- evaluation/             retrieval, answer, language and citation checks", INK, "cour"),
        ("|-- requirements.txt        Python dependencies", INK, "cour"),
        ("|-- .env.example            API-key template", INK, "cour"),
        ("`-- README.md               setup, API and data instructions", INK, "cour"),
    ]

    rect = fitz.Rect(MARGIN, y, W - MARGIN, y + 351)
    rounded_box(page, rect, PALE, LINE, 8)
    ty = y + 20
    for line, color, font in tree:
        draw_text(page, MARGIN + 16, ty, line, 7.8, color, font)
        ty += 15.6

    y = rect.y1 + 26
    section_label(page, MARGIN, y, "Responsibilities at a glance")
    y += 17
    cards = [
        ("Web layer", "app.py + frontend/", "Receives the question, preserves recent chat context and renders the answer."),
        ("Reasoning layer", "src/rag_pipeline.py", "Applies language and safety rules, prepares context and calls the LLM."),
        ("Retrieval layer", "src/kb.py", "Combines semantic and lexical rankings using reciprocal rank fusion plus intent boosts."),
        ("Data pipeline", "src/ingest.py + pdfs_data/", "Converts reference material into the searchable local index."),
    ]
    card_w = (W - 2 * MARGIN - 12) / 2
    for index, (title, files, body) in enumerate(cards):
        col, row = index % 2, index // 2
        x = MARGIN + col * (card_w + 12)
        cy = y + row * 98
        rect = fitz.Rect(x, cy, x + card_w, cy + 85)
        rounded_box(page, rect, WHITE, LINE, 7)
        draw_text(page, x + 12, cy + 21, title, 10, GREEN, "hebo")
        draw_text(page, x + 12, cy + 38, files, 7.8, GREEN_2, "cour")
        draw_wrapped(page, x + 12, cy + 56, body, card_w - 24, 7.8, 10.5, MUTED)


def page_five(doc):
    page = doc.new_page(width=W, height=H)
    page_header(page, 5, "Practical guide")
    y = heading(
        page, 78, "How the pieces work together",
        "There are two distinct workflows: rebuilding the knowledge base and serving questions.",
    )

    workflows = [
        ("A. Build or refresh the knowledge base", [
            "Place official PDF files in pdfs_data/; keep authoritative text sidecars in pdfs_data/transcribed/.",
            "Run python src/ingest.py.",
            "PyMuPDF extracts pages; recurring boilerplate and near-empty content are removed.",
            "Text is split into overlapping chunks and embedded locally with multilingual-e5-small.",
            "The resulting JSON metadata and NumPy vectors are persisted in data/vector_store/.",
        ]),
        ("B. Run and use the assistant", [
            "Start uvicorn app:app --reload and open http://127.0.0.1:8000.",
            "FastAPI warms the local retriever, then serves the browser interface and REST API.",
            "POST /api/chat receives a question plus recent history and runs the RAG pipeline in a worker thread.",
            "The pipeline retrieves five chunks by default, asks Groq for a grounded answer, checks language rules and returns sources.",
            "The same answer_question function is also available through src/chat_cli.py.",
        ]),
    ]

    for title, items in workflows:
        rect_h = 205
        rect = fitz.Rect(MARGIN, y, W - MARGIN, y + rect_h)
        rounded_box(page, rect, WHITE, LINE, 9)
        page.draw_rect(fitz.Rect(MARGIN, y, MARGIN + 7, y + rect_h),
                       color=GREEN_2, fill=GREEN_2, radius=0.08)
        draw_text(page, MARGIN + 22, y + 28, title, 13, GREEN, "hebo")
        by = y + 53
        for idx, item in enumerate(items, 1):
            page.draw_circle((MARGIN + 32, by - 3), 9, color=MINT, fill=MINT)
            draw_text(page, MARGIN + 29.2, by, str(idx), 7.5, GREEN, "hebo")
            by = draw_wrapped(page, MARGIN + 49, by, item, rect.width - 73, 8.8, 12.5, INK)
            by += 7
        y = rect.y1 + 19

    section_label(page, MARGIN, y, "Useful entry points")
    y += 17
    entries = [
        ("Web app", "app.py"),
        ("Core Q&A", "src/rag_pipeline.py"),
        ("Retrieval/index", "src/kb.py"),
        ("Re-index command", "src/ingest.py"),
        ("Automated checks", "evaluation/evaluate.py"),
    ]
    x = MARGIN
    for label, file in entries:
        width = 96
        rect = fitz.Rect(x, y, x + width, y + 58)
        rounded_box(page, rect, PALE, LINE, 6)
        draw_text(page, x + 9, y + 20, label, 7.8, MUTED, "hebo")
        draw_wrapped(page, x + 9, y + 39, file, width - 18, 7.4, 10, GREEN, "cour")
        x += width + 5

    y += 86
    draw_text(page, MARGIN, y, "Bottom line", 15, GREEN, "hebo")
    draw_wrapped(
        page, MARGIN, y + 24,
        "The project is deliberately small and local-first: documents and embeddings stay on disk, retrieval runs on the user's machine, and only the selected context plus the conversation prompt is sent to Groq for answer generation.",
        W - 2 * MARGIN, 10.5, 15, INK,
    )


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for builder in (page_one, page_two, page_three, page_four, page_five):
        builder(doc)
    metadata = {
        "title": "NADRA Guide RAG Assistant - Project Overview",
        "author": "Generated from the NADRA-RAG-main repository",
        "subject": "Purpose, technology stack, project structure and runtime flow",
        "keywords": "NADRA, RAG, FastAPI, Groq, LangChain, multilingual",
    }
    doc.set_metadata(metadata)
    doc.save(OUTPUT, garbage=4, deflate=True)
    doc.close()
    print(OUTPUT)


if __name__ == "__main__":
    main()
