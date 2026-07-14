# NADRA RAG — Implementation Analysis (2026-07-07)

Analysis of the current pipeline, prompted by two complaints: the data feels
"all over the place" and retrieval is slow. **Both are confirmed and measured
below.** TLDR: 47% of stored chunks contain garbled text, and the slowness is
~100s of one-time model loading — the actual vector search takes 0.03s.

---

## 1. How the system works

Standard three-stage RAG pipeline:

1. **`src/ingest.py`** (run once): loads the 26 PDFs in `pdfs_data/` with
   `PyPDFLoader`, tags each page with `source` + `service` metadata
   (CNIC / NICOP / POC / CRC / FRC / General), splits into ~1000-char chunks
   (150 overlap) with `RecursiveCharacterTextSplitter`, drops chunks under
   60 chars, embeds them, and persists **384 chunks** to ChromaDB at
   `data/chroma_db`.
2. **`src/rag_pipeline.py`** (per question): embeds the question with the same
   model, pulls the top-5 most similar chunks from Chroma, inserts them into a
   strict grounding system prompt (answer only from context, refuse otherwise,
   cite sources), and calls Llama 3.3 70B on Groq (`temperature=0`).
3. **`src/chat_cli.py`** and **`app.py`** are thin terminal/Streamlit
   front-ends over `answer_question()`.

## 2. Embeddings: what and why

- Model: **`sentence-transformers/all-MiniLM-L6-v2`** — 22M params, 384-dim,
  local CPU.
- Why chosen: the project's hard constraint is a completely free stack — no
  API key, no usage limits, standard baseline for English semantic search.
  For English FAQ retrieval over a few hundred chunks it is a reasonable
  choice; the embedding model is **not** the main problem.
- Real weakness: it is **English-only**, and the NADRA PDFs are bilingual
  (English + Urdu) — which feeds directly into Finding 1.

## 3. Finding 1 — the stored data is dirty (confirmed)

Measured directly from `data/chroma_db/chroma.sqlite3`:

- **181 / 384 chunks (47%) contain `�` replacement characters.** pypdf cannot
  decode the Urdu font encoding, so Urdu comes out as noise like
  `��� -�� �� ��` interleaved with the English. Almost half the embeddings
  were computed over part-English, part-garbage text, which degrades
  retrieval accuracy.
- **Two PDFs contributed zero chunks** — `nicop-complete-form-with-instruction.pdf`
  and `poc-complete-form-with-instruction.pdf` are scanned images. There is
  currently no NICOP/POC form-instruction content in the knowledge base.
- **Skew**: `registration-policy-6-0-1-english.pdf` alone accounts for
  **120 / 384 chunks (31%)**, giving generic policy text a lot of surface area
  to crowd out specific service guides in top-5 results.
- Chunks carry repeated per-page headers ("User Guideline — Smart ID
  Modification…") — more embedding noise.
- Chunk length stats: min 64 / median 546 / max 999 chars.

### Recommended fix (highest impact on answer quality)

Add a cleaning step in `ingest.py` before splitting, then re-ingest:

1. Strip garbled non-Latin sequences (Urdu is explicitly out of scope per
   `PLAN.md`).
2. Collapse whitespace; drop repeated page headers/footers.
3. Optionally swap `pypdf` for **PyMuPDF** (`pymupdf`) — better and faster
   extraction on these layouts.
4. (Later, if Urdu support becomes a goal: proper Urdu extraction +
   `intfloat/multilingual-e5-small`.)

## 4. Finding 2 — "slow retrieval" is actually slow startup (measured)

Timed on this machine (Windows 11, Python 3.14 venv):

| Stage | Time |
|---|---|
| Importing `langchain_huggingface` (pulls in PyTorch) | **72 s** |
| Loading the MiniLM model | **26 s** |
| Opening Chroma | 1 s |
| First similarity search | 0.68 s |
| Every search after that | **0.03 s** |

The vector search is essentially instant. The ~100 s import/model-load cost is
paid once per process, and because `rag_pipeline.py` lazy-loads the model
inside `_get_vectorstore()`, it lands on the **first question** — which feels
like slow retrieval. (Windows Defender scanning torch's DLLs on import is a
large part of the 72 s.)

### Recommended fixes (in order of impact)

1. **Swap the torch-based embedder for ONNX.** `fastembed`
   (`FastEmbedEmbeddings`) or Chroma's built-in default embedder runs the
   *same* all-MiniLM-L6-v2 model via ONNX Runtime — identical vectors,
   seconds to import, no 26 s model load. Also removes the heavy
   `sentence-transformers` / torch dependency.
2. **Warm up at startup**: call `_get_vectorstore()` + one dummy search when
   the CLI / Streamlit app launches, so the wait happens behind a "starting"
   message instead of on the user's first question.

## 5. Overall assessment

The architecture is sound and appropriately simple: clean ingest/pipeline/UI
separation, correct singleton reuse, a good strict grounding prompt, sensible
chunking parameters. It is a well-designed system fed dirty data and carrying
an unnecessarily heavy embedding runtime.

**Action list:**
- [ ] Clean extracted text in `ingest.py` (strip garbled Urdu, headers) and re-index
- [ ] Consider PyMuPDF instead of pypdf
- [ ] Swap `HuggingFaceEmbeddings` → ONNX embedder (fastembed)
- [ ] Warm up model/vectorstore at app startup
- [ ] Decide on OCR (pytesseract) vs. dropping the two scanned form PDFs

---

## 6. Follow-up investigation (same day): Urdu duplication, per-PDF classification, images

### Urdu duplicates the English — confirmed
Extracted pages with PyMuPDF and compared side by side: the Urdu text in the
bilingual guides is a **direct translation of the English on the same page**
(verified in new-nicop.pdf). Every bilingual PDF has full English coverage
(healthy Latin char counts across the corpus). **Stripping Urdu loses no
information** for the English-only Phase 1.

### Per-PDF extraction classification (via PyMuPDF script scan)
- **CLEAN-URDU (5)**: new-nicop, appointment-scheduling, application-tracking,
  id-cancellation-death, id-cancellation-surrender — Urdu extracts as real
  Unicode text.
- **BROKEN-URDU / MIXED (13)**: most guides (cnic-modification*, new-smart-poc,
  frc-guide-v2, birth/death-registration, payment-v4, renewal, reprint,
  proof-of-life, upload-document, new-crc, conversion-to-smartid,
  fingerprint-v11, photograph-v4) — embedded Urdu fonts have broken
  ToUnicode maps; **Urdu is unrecoverable as text** from these files. This is
  the source of the `�` garbage.
- **ENGLISH-ONLY / SCANNED (6)**: registration-policy, office-locations,
  photo-guidelines, fingerprint-guidelines, and the two scanned form PDFs
  (0 text chars, 15 images each, 3 pages each).

### Strategy for Urdu support (future phase)
Full Urdu extraction is impossible for 13/26 PDFs — but unnecessary, since
Urdu mirrors English. Use **cross-lingual RAG**: English-only knowledge base +
multilingual embeddings (`intfloat/multilingual-e5-small`, free/local, also
available in fastembed) so Urdu questions retrieve English chunks + prompt the
LLM to answer in the user's language. No OCR, no Urdu ingestion; ~10-line change.

### Images
- **Two scanned form PDFs**: options = (a) one-time vision-LLM transcription
  at ingest (render pages to PNG via PyMuPDF → Groq vision model, e.g.
  Llama 4 Scout → save .txt sidecar → ingest) — recommended, 6 pages total;
  (b) pytesseract OCR (local, weak for Urdu); (c) drop them (content overlaps
  the new-nicop / new-smart-poc guides).
- **App screenshots inside guides**: the info is already in adjacent text
  captions — do NOT OCR these, it would add noise.

### Agreed next-step order
1. Rewrite ingestion with PyMuPDF + cleaning (Latin-script only, strip
   repeated headers) → re-index. Biggest quality win.
2. Swap embedder to fastembed/ONNX + warm-up at app start (kills ~100s wait).
3. Transcribe the 2 scanned forms via vision LLM; ingest as text.
4. Build M3 evaluation set (40–60 Qs) and measure before adding retrieval
   complexity (MMR / filtering / query rewriting only if eval shows need).
5. Urdu phase later: multilingual-e5-small + answer-in-user's-language prompt.

Note: `pymupdf` is now installed in the venv (used for this investigation).

---

## 7. Implementation results (2026-07-07, same day)

All planned fixes were implemented and verified end-to-end.

### What changed
- **`src/kb.py` (new)**: shared config (paths, collection, embedding model) +
  `FastEmbedEmbeddings` (same all-MiniLM-L6-v2, ONNX runtime instead of torch;
  cache pinned to `data/fastembed_cache`) + `HybridRetriever` (vector + BM25
  with Reciprocal Rank Fusion; tokenizer does stopword removal and crude
  suffix stemming so "renew" matches "renewal").
- **`src/ingest.py` (rewritten)**: PyMuPDF extraction; keeps Latin-script text
  only (drops Urdu + broken-font glyphs incl. Latin-mapped junk like "nj");
  drops table-of-contents dot-leader lines; removes running headers/footers
  (digit-insensitive line matching across pages); ingests vision
  transcriptions from `pdfs_data/transcribed/`.
- **`src/transcribe_forms.py` (new)**: one-time transcription of the two
  scanned form PDFs via Groq vision (llama-4-scout), rendered at 150 dpi,
  output with `[Page N]` markers. Already run; sidecars committed to
  `pdfs_data/transcribed/`.
- **`src/rag_pipeline.py`**: uses `kb.get_retriever()` (hybrid); `warm_up()`
  added; TOP_K 5→6; prompt rule 2b added (answer partially when context
  partially covers the question — was causing blanket refusals).
- **`src/chat_cli.py` / `app.py`**: warm-up at startup (Streamlit via
  `st.cache_resource`); CLI stdout forced to UTF-8 (Windows cp1252 crash).
- **`requirements.txt`**: dropped langchain-community/-huggingface,
  sentence-transformers, pypdf; added fastembed, rank-bm25, pymupdf,
  langchain-text-splitters. (Old torch/sentence-transformers packages are
  still installed in the venv; uninstall to reclaim ~2–3 GB if wanted.)

### Verified results
| Metric | Before | After |
|---|---|---|
| Chunks with garbled text | 181/384 (47%) | **0/356** |
| Import + model load (first question wait) | ~100 s | **~8 s** |
| Per-question latency (retrieval + Groq) | — | 0.3–2.4 s |
| Scanned-form content in KB | none | 6 transcribed pages |

Spot checks: CNIC renewal documents (answered from registration-policy p16,
resident vs non-resident), appointment booking, payment methods, POC
eligibility, Lahore office locations — all correct with citations. Passport
trap question correctly refused. Fee questions mostly refuse because **the
corpus contains almost no fee amounts** (verified: only 2 chunks with any
currency figure) — that is correct grounding behavior, not a retrieval bug.

Embedding sanity check: fastembed vectors are identical to the old torch
vectors (cosine 1.0000), so the speed fix changed nothing semantically.

### Known limitations
- Fine-grained form-field questions ("what are the blood group options on the
  NICOP form?") rank poorly: the answer sits inside a chunk listing ~40 form
  fields, diluting both BM25 and vector signals (ranks ~36/46). Would need
  smaller/sectioned chunks or a reranker; not worth it for typical citizen
  questions.
- Corpus has no fee schedule; consider adding an official NADRA fee document
  to `pdfs_data/` if fee questions matter.
- Next milestone per PLAN.md: M3 evaluation set (40–60 questions) to measure
  hit rate / refusal accuracy before further retrieval tuning.
