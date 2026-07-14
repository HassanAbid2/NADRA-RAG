# NADRA RAG FAQ System — Project Plan (Phase 1)

**Goal:** An AI question-answering system for NADRA's five services (CNIC, NICOP, POC, CRC, FRC) built with Retrieval-Augmented Generation, grounded strictly in the official NADRA PDFs, evaluated for accuracy. 100% free stack.

## Tech Stack (all free)

| Component | Choice | Why |
|---|---|---|
| Language | Python 3.14 (already installed) | — |
| Framework | LangChain | Prebuilt loaders, splitters, retrieval chains |
| PDF text extraction | `pypdf` via LangChain's `PyPDFLoader` | Free, local |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, CPU) | Free, unlimited, no API key, good quality for English FAQ retrieval |
| Vector database | ChromaDB (local, persisted to disk) | Free, no server needed |
| LLM | Groq free tier — Llama 3.3 70B (`llama-3.3-70b-versatile`) | Free API key, no credit card, fast, generous daily limits |
| Demo UI | Streamlit (local) | Free, ~50 lines for a chat page |

**Only external dependency:** a free Groq API key from https://console.groq.com (sign up, create key, put in `.env`).

## Project Structure

```
NADRA RAG/
├── pdfs_data/               # extracted from pdfs_data.zip (25 official PDFs)
├── data/chroma_db/          # persisted vector store (generated)
├── src/
│   ├── ingest.py            # PDF → text → chunks → embeddings → ChromaDB
│   ├── rag_pipeline.py      # retriever + prompt + Groq LLM chain
│   └── chat_cli.py          # terminal Q&A loop with source citations
├── evaluation/
│   ├── test_questions.json  # gold Q&A test set (~40–60 questions)
│   └── evaluate.py          # retrieval hit-rate + answer grading
├── app.py                   # Streamlit demo chat UI
├── .env                     # GROQ_API_KEY (never commit)
├── requirements.txt
└── PLAN.md
```

## Milestones

### M1 — Setup & Knowledge Base Ingestion
1. Unzip `pdfs_data.zip` (skip `__MACOSX` junk); include `NADRA_Office_Locations_Pakistan.pdf`.
2. Create venv, install: `langchain langchain-groq langchain-community langchain-huggingface langchain-chroma chromadb sentence-transformers pypdf streamlit python-dotenv`.
3. `ingest.py`: load each PDF → split with `RecursiveCharacterTextSplitter` (~1000 chars, ~150 overlap) → attach metadata (source file, page, service category) → embed → persist to ChromaDB.
4. Sanity check: inspect chunks from a few PDFs (some may be scanned/image-heavy — verify text actually extracted; fall back to OCR via free `pytesseract` only if needed).

### M2 — RAG Pipeline
5. `rag_pipeline.py`: retriever (top-k ≈ 4–6 chunks) + strict grounding prompt:
   - Answer ONLY from provided context.
   - If the answer isn't in the context, say so explicitly ("I don't have verified NADRA information on that") — critical government-context requirement from the problem statement.
   - Cite source document + page in answers.
6. `chat_cli.py`: interactive terminal loop for manual testing.

### M3 — Evaluation (explicit Phase 1 requirement)
7. Build `test_questions.json`: ~40–60 questions across all 5 services covering documents required, eligibility, fees, timelines, renewal — plus **out-of-scope trap questions** (e.g. "How do I renew my passport?") that the system must refuse.
8. `evaluate.py` measures:
   - **Retrieval hit rate** — is the correct source chunk in the top-k?
   - **Answer correctness** — LLM-as-judge (also free via Groq) comparing answer vs. gold reference, plus spot manual review.
   - **Refusal accuracy** — % of out-of-scope questions correctly declined.
9. Iterate: tune chunk size, top-k, prompt based on failures. Document results.

### M4 — Demo UI & Wrap-up
10. `app.py`: Streamlit chat page with answer + expandable source citations.
11. `README.md`: setup instructions, architecture diagram, evaluation results — deliverable documentation for the internship.

## Free-tier limits to keep in mind
- Groq free tier: roughly 1,000+ requests/day and ~12K tokens/min for Llama 3.3 70B — ample for dev + evaluation. If a rate limit hits during batch evaluation, add a small delay between calls.
- Everything else (embeddings, vector DB, UI) runs locally with no limits.

## Out of scope (per problem statement)
Live application tracking, complaint handling, Urdu (stretch goal later), NADRA internal system integration, public deployment.
