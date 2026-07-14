# Code Changes — Exactly What Changed, Where, and Why

*Companion to `IMPROVEMENTS.md` (the story) and `ANALYSIS.md` (the findings).
This file maps every change to its file and function, with the problem it
caused and why the fix works.*

---

## 1. `src/kb.py` — NEW FILE

### 1.1 Shared configuration (top of file)
**Before:** `EMBEDDING_MODEL`, `CHROMA_DIR`, and the collection name were
declared twice — once in `ingest.py`, once in `rag_pipeline.py`.
**Problem:** if one file changed and the other didn't, the query embeddings
would come from a different model than the stored ones and every search would
silently return garbage. Nothing would error — results would just be wrong.
**Fix:** one module owns the paths, collection name, and model name; both the
writer (`ingest.py`) and the reader (`rag_pipeline.py`) import from it, so
they physically cannot drift apart.

### 1.2 `FastEmbedEmbeddings` class
**Before:** `HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")`
— the PyTorch implementation.
**Problem (measured):** importing the torch stack took 72 s and loading the
model 26 s on this machine, ~100 s total, paid on the *first question* because
loading was lazy. This is what felt like "slow retrieval" — the actual search
took 0.03 s.
**Fix:** a ~15-line adapter that runs the *same* model through `fastembed`
(ONNX Runtime). Import drops to ~4 s, model load to ~1 s.
**Why it's safe:** verified by embedding identical sentences with both
implementations — cosine similarity 1.0000 on every test. The vectors are
mathematically identical; only the runtime changed.

```python
class FastEmbedEmbeddings(Embeddings):
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        cache = PROJECT_ROOT / "data" / "fastembed_cache"
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache))
```

The `cache_dir` line exists because fastembed's default cache is the Windows
Temp folder, which the OS may wipe — that would force a silent ~90 MB
re-download later. Pinning it under `data/` makes it permanent.

### 1.3 `HybridRetriever` class
**Before:** retrieval was `vectorstore.similarity_search(question, k=5)` —
pure vector search.
**Problem (observed):** for "What documents do I need to renew my CNIC?" the
top-5 chunks were all blank legal undertaking annexes; the actual answer
(registration policy p16) existed in the store but was never retrieved.
Embeddings capture paraphrase well but underweight rare exact terms
(acronyms, form-field names, "fee").
**Fix:** run **two** searches — vector similarity *and* BM25 keyword scoring —
then merge the rankings with Reciprocal Rank Fusion:

```python
score = sum(1.0 / (60 + rank) for each ranking the chunk appears in)
```

A chunk wins by ranking near the top of *either* list. Both searches
over-fetch (`fetch_k = max(4*k, 20)`) so fusion has real candidates to work
with. BM25 zero-score documents are excluded so keyword noise can't vote.

### 1.4 `_tokenize()` — stopwords + crude stemming
**Before (first attempt):** `re.findall(r"[a-z0-9]+", text.lower())`.
**Problem (observed):** BM25 matches literally. The question says "renew";
the policy document says "renewal" — no match. "Documents" ≠ "document".
The renewal question still failed even after adding BM25.
**Fix:** strip a fixed list of suffixes ("ments", "ment", "tion", "ing",
"al", "s", …) so both query and document reduce to the same root
("renewal" → "renew", "documents"/"document" → "docu"), and drop ~40
stopwords ("the", "what", "how") so they stop consuming BM25's scoring
budget. The stems don't need to be linguistically correct — only
*consistent*, because the same function processes both sides.

---

## 2. `src/ingest.py` — REWRITTEN

### 2.1 `PyPDFLoader` → PyMuPDF (`fitz`) in `load_pdf_documents()`
**Problem:** pypdf mangled the Urdu fonts in these PDFs into `�` replacement
characters — 181 of 384 stored chunks (47%) contained garbage interleaved
with the English, dragging every embedding in the wrong direction.
**Why PyMuPDF:** on the 5 PDFs with intact Urdu fonts it extracts real
Unicode; on all PDFs its text ordering and speed are better. (On the 13 PDFs
with broken font maps *no* extractor can recover the Urdu — which is fine,
because the Urdu is a translation of the English on the same page, verified
side-by-side.)

### 2.2 `clean_lines()` — the Latin-only filter
**Problem:** even with better extraction, Urdu text (readable or broken) is
noise for an English-only knowledge base and an English-only embedding model.
**Fix, in order:**
1. `unicodedata.normalize("NFKC", text)` — normalizes ligatures/width variants.
2. `_NON_LATIN` regex replaces every character outside printable ASCII +
   common typography (`‘’“”–—•·…`) with a space — deletes Urdu script and
   broken-font glyphs in one pass.
3. A keep-rule per line (see 2.3).

### 2.3 The line keep-rule — three iterations
This rule decides which cleaned lines survive. It took three attempts,
each driven by inspecting the actual stored chunks afterwards:

| Version | Rule | What went wrong |
|---|---|---|
| 1 | keep if ≥1 letter or ≥2 digits | kept two-letter junk lines like "nj" |
| 2 | keep if ≥3 letters total or ≥2 digits | still kept "nj ( ) nj" — four letters *total* |
| 3 (final) | keep if it contains a real **word** — `[A-Za-z]{3,}` consecutive — or ≥2 digits | 0 junk chunks |

**Why "nj" existed at all:** some broken Urdu ligatures map into *Latin*
letters, not `�`, so they survived the character filter. They never form
3+ consecutive letters, so requiring an actual word kills them. The
digit clause keeps genuinely numeric lines (fees "750/-", helpline "1777",
dates).

There is also an explicit scrub for the two known junk shapes before the
keep-rule runs: `re.sub(r"\bnj\b|\(\s*\)", " ", line)` — this removes junk
*embedded inside* otherwise-good lines, which the line-level rule can't drop.

### 2.4 TOC filter
```python
if re.search(r"\.{4,}", line):  # "Fees ............ 12"
    continue
```
**Problem (observed):** the registration policy's table-of-contents chunks
("Attestation/Verification of CNICF ..............") ranked *top-2* for real
questions — dot-leader lines are keyword-dense and content-free.
**Fix:** any line with 4+ consecutive dots is a TOC entry; drop it.

### 2.5 `remove_boilerplate()` + `_boiler_key()`
**Problem:** every page of every guide repeats its running header ("User
Guideline — Smart ID Modification — Non-Printable"), which ended up inside
chunks as embedding noise.
**First attempt:** drop lines that repeat verbatim on ≥60% of a document's
pages. **It missed most headers**, because headers contain the page number —
"User Guideline 7" and "User Guideline 13" are different strings.
**Fix:** compare lines with digits stripped out first:

```python
def _boiler_key(line):
    return re.sub(r"\s+", " ", re.sub(r"\d+", "", line)).strip()
```

Guards: only documents with ≥4 pages, only lines ≤60 chars — so a short real
sentence repeated in a 3-page form can't be deleted by accident.

### 2.6 `load_transcribed_documents()` — new
**Problem:** two PDFs (`nicop-complete-form-with-instruction.pdf`,
`poc-complete-form-with-instruction.pdf`) are scans — zero extractable text.
The knowledge base had *no* NICOP/POC form-instruction content at all.
**Fix:** ingest also reads `pdfs_data/transcribed/*.txt` (produced by
`transcribe_forms.py`, below), splitting on `[Page N]` markers so each page
gets correct `page` metadata for citations.

### 2.7 Page numbers now 1-based
`enumerate(pdf, start=1)` — pypdf's loader used 0-based page metadata, so
citations pointed one page off from what a human sees in a PDF viewer.

---

## 3. `src/transcribe_forms.py` — NEW FILE

**Problem:** see 2.6 — scanned forms, no text layer, 15 images per file.
**Fix:** one-time script that renders each page at 150 dpi
(`page.get_pixmap(dpi=150)`), base64-encodes the PNG, and sends it to a
vision model on Groq (`meta-llama/llama-4-scout-17b-16e-instruct`) with a
prompt to transcribe the English and skip the Urdu. Output goes to
`pdfs_data/transcribed/<stem>.txt` with `[Page N]` markers.
**Why vision-LLM over OCR:** only 6 pages total, so cost is trivial even on
a free tier; it preserves structure (field lists, notes) far better than
Tesseract, whose Urdu handling is also weak. **Why a sidecar file:** the
transcription runs once and is cached on disk — re-running ingestion never
re-calls the vision API (`if out_path.exists(): skip`).

---

## 4. `src/rag_pipeline.py`

### 4.1 Deleted the local embedding/vectorstore singletons
`_get_vectorstore()` and its `HuggingFaceEmbeddings` are gone; the module now
imports `kb` and uses `kb.get_retriever()`. See 1.1/1.2 for why.

### 4.2 `retrieve()` — now hybrid
```python
return kb.get_retriever().search(question, k=k)   # was: similarity_search
```
See 1.3 for the problem this solves.

### 4.3 `warm_up()` — new
**Problem:** all loading was lazy, so the entire startup cost landed on the
user's first question.
**Fix:** one function loads the embedding model, opens Chroma, builds the
BM25 index, and runs a dummy search. Both front-ends call it at launch, so
the wait happens behind a "Loading…" message instead of inside the first
answer. (Building BM25 also *requires* startup work — it reads all 356
chunks out of Chroma — so warm-up became structural, not just cosmetic.)

### 4.4 `TOP_K = 5` → `6`
The cleaned chunks are shorter (median 372 chars vs 546 before), so six
chunks fit comfortably in context and give the LLM one more shot at
containing the answer.

### 4.5 System prompt — rule 2 split into 2 / 2b
**Before:** "If the context does not contain the answer, reply exactly
[refusal]."
**Problem (observed):** the model read this as all-or-nothing and refused
*every* test question, even with the correct chunk in context, whenever any
part of the question wasn't fully covered.
**After:** rule 2 fires only when there is *no* relevant information; new
rule 2b instructs: if the context answers the question *partially*, give the
covered part and say explicitly what official documents don't cover.
**Why it's still safe:** the model remains forbidden from using outside
knowledge; it only gained permission to share verified partial information
instead of withholding it.

---

## 5. `src/chat_cli.py`

### 5.1 `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`
**Problem:** Windows consoles can default to cp1252; answers contain
typographic characters (’, —), which made `print()` crash with
`UnicodeEncodeError` — observed during testing.
**Fix:** force UTF-8 with graceful degradation.

### 5.2 Warm-up at launch
Prints "Loading knowledge base …" → `warm_up()` → "Ready." before the input
loop. See 4.3.

## 6. `app.py`

```python
@st.cache_resource(show_spinner="Loading knowledge base (one-time) ...")
def _warm_up_once():
    warm_up()
```
**Why `st.cache_resource`:** Streamlit re-executes the whole script on every
user interaction. Without the decorator, warm-up code at module level would
still be *checked* every rerun; with it, Streamlit guarantees one execution
per server process and shows a spinner during that first load.

## 7. `requirements.txt`

| Removed | Why |
|---|---|
| `sentence-transformers`, `langchain-huggingface` | torch stack replaced by fastembed (1.2) |
| `pypdf`, `langchain-community` | replaced by PyMuPDF (2.1); nothing imports langchain-community anymore |
| `langchain` | no direct imports remain; sub-packages are listed explicitly |

| Added | Why |
|---|---|
| `fastembed` | ONNX embeddings (1.2) |
| `rank-bm25` | keyword half of hybrid retrieval (1.3) |
| `pymupdf` | PDF extraction + page rendering (2.1, 3) |
| `langchain-text-splitters` | was an implicit dependency; now explicit |

The old torch packages are still *installed* in the venv (uninstalling wasn't
necessary); a fresh `pip install -r requirements.txt` on another machine gets
only the lean stack.

---

## Net effect

| | Before | After |
|---|---|---|
| Garbled chunks | 181/384 (47%) | 0/356 |
| First-answer wait | ~100 s | ~8 s |
| "Renew CNIC" question | refused (wrong chunks retrieved) | exact answer, cited to policy p16 |
| Scanned-form content | absent | 6 pages, searchable |
| Trap questions | refused ✔ | still refused ✔ |
