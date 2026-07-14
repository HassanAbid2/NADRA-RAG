# How the NADRA RAG Got Better — Problems, Experiments, and Fixes

*Written 2026-07-07. This is the story of the debugging session: what was wrong,
how each problem was found, what was tried, and what finally fixed it. The
clinical version with tables lives in `ANALYSIS.md`; this one explains the
reasoning.*

---

## Where we started

The system looked fine on the surface. Three clean stages: `ingest.py` turned
26 official NADRA PDFs into ~1000-character chunks and stored them in a local
ChromaDB vector database; `rag_pipeline.py` took a question, found the 5 most
similar chunks, and asked Llama 3.3 70B (on Groq) to answer strictly from
them; a CLI and a Streamlit page sat on top.

But two things felt off: answers were slow to arrive, and the underlying data
"felt all over the place." Both feelings turned out to be measurably correct —
just not for the reasons you might guess.

---

## Problem 1: "Retrieval is slow"

**The experiment.** Instead of assuming, I put a stopwatch on every stage of
answering a question:

| Stage | Time |
|---|---|
| Importing the embedding library (which pulls in PyTorch) | 72 s |
| Loading the MiniLM model into memory | 26 s |
| Opening ChromaDB | 1 s |
| The actual vector search | **0.03 s** |

**The discovery.** Retrieval was never slow. The search itself takes 30
milliseconds. What hurt was ~100 seconds of *one-time startup cost* — and
because the code lazily loaded the model on first use, that cost landed
exactly when you typed your first question. It felt like slow retrieval; it
was actually slow importing. (A big chunk of those 72 seconds is Windows
Defender scanning PyTorch's DLLs every time Python imports them.)

**The fix.** The all-MiniLM-L6-v2 embedding model doesn't need PyTorch — the
same model exists in ONNX format, a lean format for running (not training)
neural networks. The `fastembed` library runs it with a 4-second import and
near-instant model load.

**The proof.** Before trusting it, I embedded identical sentences with both
the old PyTorch version and the new ONNX version and compared the vectors:
**cosine similarity 1.0000 on every test sentence** — mathematically the same
embeddings, so nothing about search quality changed. Only the wait died.

On top of that, both the CLI and the Streamlit app now call `warm_up()` at
launch, so whatever loading remains happens behind a "Loading knowledge
base..." message instead of on your first question.

**Result: ~100 s → ~8 s to first answer; each question takes 0.3–2.4 s
(most of which is Groq generating the answer, not retrieval).**

---

## Problem 2: The data really was "all over the place"

**The experiment.** I opened the ChromaDB SQLite file directly and read the
stored chunks — not the PDFs, the actual text the system searches over.

**The discovery — worse than expected.** 181 of 384 chunks (47%) contained `�`
replacement characters surrounded by garbage. The cause: NADRA's guides are
bilingual, English and Urdu side by side, and the old PDF reader (`pypdf`)
choked on the Urdu fonts. Almost half of every "page" the system had memorized
was part English, part noise — and the noise was baked into the embeddings,
quietly dragging every similarity score in the wrong direction.

**A key question along the way: is the Urdu unique content, or a translation?**
This mattered enormously. If the Urdu said things the English didn't, we'd
have to recover it somehow. So I extracted pages with a better library
(PyMuPDF) and read the two languages side by side. Verdict: **the Urdu is a
direct translation of the English on the same page.** Deleting it loses
nothing. Even better, the scan revealed that in 13 of the 26 PDFs the Urdu is
*unrecoverable anyway* — the fonts inside those files have broken character
maps, so no extractor on earth gets readable Urdu out of them. (When Urdu
support becomes a goal, the right move is multilingual embeddings so Urdu
*questions* can find English *chunks* — not Urdu extraction.)

**The fix — a rewritten `ingest.py`.** It now:
1. Extracts with PyMuPDF (better and faster than pypdf on these layouts).
2. Keeps only Latin-script text — Urdu and broken glyphs become spaces.
3. Drops running headers/footers ("User Guideline — Smart ID Modification")
   by finding lines that repeat across most pages of a document.
4. Drops table-of-contents lines ("Fees .............. 12") — pure noise that
   was ranking highly in searches.

**The iteration nobody plans for.** The first cleaning pass wasn't enough.
Some broken Urdu ligatures don't turn into `�` — they map to *Latin letters*,
leaving fragments like `nj ( ) nj ( )` scattered through the text. My first
filter ("keep lines with 3+ letters") kept them, because `nj nj` has four
letters. The rule that worked: a real line must contain an actual **word** —
three or more *consecutive* letters — or a number. Junk fragments never form
real words. Two more rounds of ingest-inspect-tighten got the count from
181 garbled chunks → 39 fragments → 6 → **0**.

The headers needed a similar trick: "User Guideline 7" and "User Guideline 13"
are different strings, so naive repeat-detection missed them. Comparing lines
*with digits stripped out* caught them all.

---

## Problem 3: Two PDFs were invisible

The NICOP and POC application-form PDFs are scans — photographs of paper, no
text layer at all. The old ingest silently produced zero chunks from them, so
the knowledge base had literally no NICOP/POC form instructions.

**The fix.** A new one-time script, `transcribe_forms.py`, renders each page
to an image and sends it to a vision model on Groq (Llama 4 Scout) with the
instruction "transcribe all the English text, skip the Urdu." It's only 6
pages total. The transcriptions are saved as text files in
`pdfs_data/transcribed/`, and `ingest.py` picks them up like any other
document. Spot-checking the output showed clean, well-structured text — form
fields, instructions, even the fine print.

---

## Problem 4 (the surprise): retrieval quality was bad all along

This was the most interesting part. After all the cleaning, I ran an
end-to-end test with real questions — and **every single one was refused**,
including "What documents do I need to renew my CNIC?", which the documents
definitely answer.

**First suspicion: did the new embedder break something?** No — that's why the
cosine-similarity test above mattered. Identical vectors. The embedder was
innocent.

**So I looked at what retrieval actually returned.** For the CNIC renewal
question, the top 5 chunks were all legal *undertaking annexes* — blank
declaration forms full of underscores and the word "CNIC" repeated. The actual
answer (registration policy, page 16: for residents, "No other document will
be required") existed in the database but never got retrieved. The dirty data
had been *hiding* this weakness the whole time; the old system had it too.

**Why pure vector search failed here.** Embeddings are great at meaning
("get a new card" ≈ "renewal") but surprisingly bad at rare exact terms —
acronyms, field names, specific words like "fee". The classic complement is
**BM25**, a keyword-scoring algorithm (a smarter cousin of word counting).
So the retriever became a **hybrid**: run both searches, then merge the two
rankings with Reciprocal Rank Fusion — a simple, robust formula where a chunk
scores well if *either* method ranks it near the top.

**The second bug hiding inside the first.** Even with BM25, the renewal
question failed — because BM25 matches words *literally*. The question says
"renew"; the document says "renewal". "Documents" ≠ "document". The fix was a
small stemmer in the tokenizer: strip common suffixes so both sides reduce to
the same root ("renew", "docu"), plus a stopword list so words like "the" and
"what" stop wasting the scoring budget.

**The third fix was in the prompt, not the code.** The system prompt demanded
all-or-nothing: answer fully from context or refuse entirely. The model chose
refusal constantly. One added rule — *"if the context answers only part of the
question, give the part that IS covered and say what's missing"* — plus
raising the number of retrieved chunks from 5 to 6, and answers started
flowing.

**And one non-bug worth knowing about.** Fee questions still get refused —
correctly. I searched the entire knowledge base: only 2 chunks in the whole
corpus mention any currency amount. NADRA's guides describe *how* to pay, not
*how much*. The system refusing to invent fee numbers is exactly the behavior
a government FAQ should have. If fees matter, the fix is adding an official
fee-schedule PDF to `pdfs_data/`, not touching the code.

**After all four fixes**, the test suite told a different story: CNIC renewal
answered precisely (resident vs. non-resident, cited to the right page),
appointment booking, payment methods, POC eligibility, Lahore office
locations — all correct with citations. And the trap question "How do I renew
my passport?" still gets refused, as it should (passports are not NADRA's
job).

---

## What's honestly still imperfect

- **Very fine-grained form questions** ("what are the blood group options on
  the NICOP form?") rank poorly. The answer sits inside a chunk listing ~40
  form fields, so its signal is diluted for both search methods (it ranked
  ~36th and ~46th). Fixable with smaller chunks or a reranking model, but
  probably not worth it for real citizen questions.
- **No fee data** in the corpus, as described above.
- **The real judge hasn't run yet**: the M3 evaluation set (40–60 gold
  questions across all five services, per `PLAN.md`) is the next milestone.
  Rule of thumb going forward: measure first, tune second — every retrieval
  change above was driven by looking at actual retrieved chunks, not by
  guessing.

## The one-line summary

The architecture was never the problem. The system was a decent design fed
47%-garbage data, carrying a 100-second PyTorch tax it didn't need, using a
retrieval method with a known blind spot for exact keywords, and a prompt that
preferred silence over partial help. Fix the data, drop the tax, hybridize the
search, soften the all-or-nothing rule — and the same architecture works.
