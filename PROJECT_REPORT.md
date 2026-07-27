# NADRA RAG FAQ Assistant — Project Report

*Prepared 2026-07-28. Covers the project from data gathering through the first
measured evaluation. Companion documents: `PLAN.md` (original plan),
`ANALYSIS.md` (measurements), `IMPROVEMENTS.md` (debugging narrative),
`CODE_CHANGES.md` (change-by-change detail).*

---

## 1. Executive summary

A document-grounded question-answering assistant for NADRA's citizen services
(CNIC, NICOP, POC, CRC, FRC), built on Retrieval-Augmented Generation over the
official NADRA PDFs. It answers in English, Urdu and Roman Urdu, cites the
source document and page for every claim, and refuses questions the official
documents do not cover.

Everything runs on a free stack: embeddings and vector search are local CPU,
and only answer generation calls an external API (Groq's free tier).

**Where it stands after the first full evaluation (54 gold questions):**

| Metric | Result |
|---|---|
| Retrieval hit rate | **48/48 (100%)** |
| Answer correctness (LLM-judged, 20 graded) | 11/20 (55%) |
| Refusal accuracy on out-of-scope traps | 3/4 graded (2 unrun — API quota) |
| Garbled text in the knowledge base | 0 of 365 chunks |
| Time to first answer | ~8 s (was ~100 s) |

Retrieval is solved. Answer generation is now the weakest layer and is the
clear next priority — details in §8.

---

## 2. What was built

Three stages, plus two front-ends:

| Stage | File | Role |
|---|---|---|
| Ingestion | [src/ingest.py](src/ingest.py) | PDFs → cleaned English text → 365 chunks → embeddings |
| Knowledge base | [src/kb.py](src/kb.py) | Embedding model, vector store, hybrid retriever |
| Answering | [src/rag_pipeline.py](src/rag_pipeline.py) | Retrieve → ground → generate → cite |
| Web app | [app.py](app.py) + [frontend/](frontend/) | FastAPI REST API + HTML/CSS/JS chat UI |
| CLI | [src/chat_cli.py](src/chat_cli.py) | Terminal loop for testing |

**Stack.** Python; PyMuPDF for extraction; `intfloat/multilingual-e5-small`
embeddings via FastEmbed (ONNX, local CPU); a local NumPy vector store plus
BM25 keyword search fused with Reciprocal Rank Fusion; Llama 3.3 70B on Groq
for generation; FastAPI + vanilla JS for the UI. No paid services.

---

## 3. Phase 1 — Gathering the data, and what went wrong

### 3.1 Finding authoritative source documents

The first problem was simply **finding data worth grounding on**. NADRA does not
publish a single machine-readable knowledge base. What exists is a scatter of
PDF user guides, a registration policy document, and application forms, spread
across the website and the Pak-ID app help pages.

We assembled 27 PDFs covering the five services plus cross-cutting topics
(payment, appointment booking, application tracking, photograph and fingerprint
guidelines, document upload, cancellation, proof of life).

### 3.2 Choosing which PDFs to keep

Several documents overlapped or came in multiple versions, which forced explicit
selection decisions:

- `fingerprint-guidelines.pdf` vs `fingerprint-guidelines-v11.pdf`
- `photo-guidelines.pdf` vs `photgraph-guidelines-v4.pdf`
- `cnic-modification.pdf` vs `cnic-modification-nonprintable-17-11-2025.pdf`

Both versions of each pair were kept: they are not exact duplicates, and the
newer file does not always supersede the older one. The cost is some redundancy
in retrieval; the benefit is no accidental loss of guidance. This remains an
open cleanup item (§9).

One document also dominated the corpus: `registration-policy-6-0-1-english.pdf`
alone produced 120 of the original 384 chunks (31%), giving generic policy text
disproportionate weight in search results.

### 3.3 Gap 1 — no office locations *(verified during this report)*

NADRA does **not** publish a downloadable list of its registration centres. Its
official locator is an interactive map/app only, so there was no document to
ingest and the assistant could not answer "where is the nearest NADRA office?"
— one of the most obvious citizen questions.

**How it was solved:** Google Maps (Google Places data) was queried for
NADRA-related government offices, and the results were compiled into
`NADRA_Office_Locations_Pakistan.pdf` (generated 2026-06-30) and ingested like
any other document. The dataset contains:

- **97 locations across 20 cities** — Karachi (10), Lahore (10), Rawalpindi (8),
  Faisalabad (8), Multan (9), Peshawar (7), Islamabad (6), Quetta (5),
  Gujranwala (5), Hyderabad (4), Bahawalpur (4), Sargodha (4), and others.
- Mega Centres, Executive Centres, Regional Head Offices and standard
  Registration Centres.
- Passport-only offices and e-Sahulat fee-collection franchises deliberately
  **excluded** — they are not part of NADRA's own registration network.
- NADRA's national helpline (051-111-786-100) substituted wherever Google Maps
  listed no branch number.

The document states its own provenance on page 1, so any answer drawn from it
carries the caveat that it is **not an official NADRA publication**, coverage
skews to major cities, and details may be stale.

**Reproducibility gap:** the script that queried Google Places was not saved
and is not in the repository. The dataset cannot currently be refreshed or
expanded without rewriting it. This should be fixed before the data ages.

### 3.4 Gap 2 — no fee information

An early check found only **2 chunks in the entire corpus containing any
currency figure**. NADRA's guides explain *how* to pay, not *how much*. Every
fee question was therefore refused — correct grounding behaviour, but useless
to a citizen.

**How it was solved:** NADRA's published fee schedule was transcribed by hand
into `pdfs_data/transcribed/nadra-fee-structure.txt`, structured as 7 labelled
pages so citations resolve to a real page number. It covers CNIC and Smart CNIC
(PKR), CRC/FRC, Smart NICOP by zone (USD), Smart POC, non-printable field
updates, multiple-ID clearance, age modification, succession certificates,
delivery charges, validity periods and the fee-deposit bank details — each with
its Normal / Urgent / Executive fee and processing timeline.

---

## 4. Phase 2 — Ingestion problems

### 4.1 Half the knowledge base was garbage

Reading the stored chunks directly showed **181 of 384 (47%) contained `�`
replacement characters**. NADRA's guides are bilingual, and the original PDF
reader (`pypdf`) could not decode the embedded Urdu fonts. Almost half of every
stored page was part English, part noise — and that noise was baked into the
embeddings, dragging every similarity score off-target.

Before deleting the Urdu, we checked whether it carried unique content:
extracting pages with PyMuPDF and comparing side by side confirmed **the Urdu is
a direct translation of the English on the same page**. Deleting it loses
nothing. In 13 of 27 PDFs the Urdu is unrecoverable anyway — the embedded fonts
have broken character maps, so no extractor can read them.

**Fix:** rewrote ingestion on PyMuPDF, keeping Latin-script text only, dropping
table-of-contents dot-leader lines and repeated running headers.

**The iteration that wasn't planned.** The first cleaning pass was not enough.
Some broken Urdu ligatures map to *Latin* letters rather than `�`, leaving
fragments like `nj ( ) nj`. A "keep lines with 3+ letters" rule kept them,
because `nj nj` has four letters. The rule that worked: a line must contain a
real **word** — three or more *consecutive* letters — or a number. Three rounds
of ingest-inspect-tighten took it from 181 garbled chunks → 39 → 6 → **0**.

Running headers needed a similar trick: "User Guideline 7" and "User Guideline
13" are different strings, so naive repeat-detection missed them. Comparing
lines with digits stripped caught them all.

### 4.2 Two PDFs were invisible

The NICOP and POC application forms are scans — photographs of paper with no
text layer. Ingestion silently produced zero chunks from them, so the knowledge
base had no NICOP/POC form instructions at all.

**Fix:** [src/transcribe_forms.py](src/transcribe_forms.py) renders each page at
150 dpi and sends it to a vision model (Llama 4 Scout on Groq) to transcribe the
English. Only 6 pages, run once, cached as text sidecars.

### 4.3 The vision model invented data *(found 2026-07-28)*

A later audit found the POC transcription contained a `### FEES` block listing
17 PKR amounts. Rendering page 1 of the source PDF showed **that page is a blank
application form with no fee section at all** — the vision model had fabricated
the entire block. Worse, it contradicted the real fee schedule (quoting NICOP in
PKR when it is priced in USD by zone), so the knowledge base held two competing
answers for the same question.

The block was removed and the corpus re-ingested. **Lesson: machine
transcription of source documents must be spot-checked against the original
page, not trusted because the output looks well-formed.**

---

## 5. Phase 3 — Retrieval problems

### 5.1 Pure vector search had a blind spot

After cleaning, an end-to-end test refused *every* question — including "What
documents do I need to renew my CNIC?", which the corpus definitely answers.
Inspecting what retrieval actually returned showed the top 5 chunks were blank
legal undertaking annexes. The real answer (registration policy p16) was in the
database but never retrieved.

Embeddings capture paraphrase well but underweight rare exact terms — acronyms,
form-field names, "fee". **Fix:** run vector search *and* BM25 keyword search,
then merge with Reciprocal Rank Fusion, so a chunk wins if *either* method ranks
it highly.

### 5.2 A second bug hiding inside the first

Even with BM25 the query failed, because BM25 matches literally: the question
says "renew", the document says "renewal"; "documents" ≠ "document". **Fix:** a
crude stemmer in the tokenizer plus a stopword list. The stems need not be
linguistically correct, only *consistent* — the same function processes both
sides.

### 5.3 Intent routing

Some questions have one obviously correct source that generic scoring missed —
fee questions belong to the fee schedule, location questions to the office list.
[src/kb.py](src/kb.py) applies small per-source boosts keyed on query intent.

This required a subtle bug fix: boost lookups compare against a **lowercased**
source name, so a boost registered under the file's mixed-case name
(`NADRA_Office_Locations_Pakistan.pdf`) never matched and the feature was
silently invisible.

### 5.4 Multilingual support

Urdu and Roman Urdu questions had to retrieve English chunks. **Fix:** swapped
the English-only embedding model for `intfloat/multilingual-e5-small`, which maps
all three into one space — no Urdu ingestion, no OCR. Supporting work: informal
city-name normalisation (`pindi` → Rawalpindi, `isb` → Islamabad), conjugated
Roman Urdu verb forms in the language classifier, and stripping source labels in
Urdu script and Roman Urdu as well as English.

---

## 6. Phase 4 — Generation and operational problems

**All-or-nothing refusals.** The system prompt demanded a full answer or a
refusal, and the model chose refusal constantly. Adding one rule — *if the
context answers part of the question, give that part and say what is missing* —
unblocked answering.

**A 100-second wait mistaken for slow retrieval.** Timing every stage showed the
actual vector search takes **0.03 s**; the pain was ~100 s of one-time PyTorch
import and model loading, landing on the user's first question because loading
was lazy. Fix: run the same model through ONNX (verified as producing
mathematically identical vectors — cosine similarity 1.0000) and warm up at
startup. **~100 s → ~8 s.**

**Groq free-tier limits.** Development repeatedly hit HTTP 429. Rotating to a
second API key did not help: **the quota is per organisation, not per key.**
Mitigations: a friendly user-facing message on 429, a trimmed system prompt, and
TOP_K tuned to 5. This limit constrained the evaluation below.

**Duplicate citations.** One document cited on pages 3, 5 and 9 rendered as three
separate source cards; citations are now collapsed to one card per document.

---

## 7. Evaluation

### 7.1 Method

A gold set of **54 questions** in [evaluation/test_questions.json](evaluation/test_questions.json),
covering all five services plus fees, locations, general processes, Roman Urdu
questions, and 6 out-of-scope traps (passport renewal, driving licence, UK visa,
weather, FBR tax filing, and a request for a named person's CNIC number).

Every reference answer was **verified against the actual corpus text** — by
dumping what retrieval returns for each question and reading the source
passages — rather than written from assumption. Each question lists the
document(s) that genuinely contain the answer; retrieval scores a hit if any of
them appears in the top-k.

Three metrics, via [evaluation/evaluate.py](evaluation/evaluate.py):

- **Retrieval hit rate** — local only, no API cost, so it runs over the whole set.
- **Refusal accuracy** — does the system decline out-of-scope questions?
- **Answer correctness** — LLM-as-judge against the gold answer.

### 7.2 Retrieval: 96% → 100%

The first run scored **46/48 (96%)**, with two failures that were real bugs:

| Question | Failure | Cause |
|---|---|---|
| "How do I change the address on my CNIC?" | returned the office-locations list | `address` was treated as a location signal, so a *modification* request was routed to the branch directory |
| "Until what age is a CRC valid?" | returned the CRC app guide | the validity table lives only in the fee schedule, but no validity keyword routed there |

Both were fixed in [src/kb.py](src/kb.py): `address` now counts as a location
signal only when no modification verb is present, and validity/expiry keywords
route to the fee schedule. Re-run: **48/48 (100%)**.

### 7.3 Generation: the weak layer

Graded on a 26-question sample (all 6 traps plus 20 answer questions spread
across services) to stay inside the free daily token budget:

- **Answer correctness: 11/20 (55%)**
- **Refusal accuracy: 3/4 graded** — the final 2 traps could not be run because
  the daily quota was exhausted mid-run.

Reviewing the 9 failed answers, roughly four look like strict grading rather than
defects — for example the new-CNIC fee answer gave the correct Rs. 0 / 1,150 /
2,150 amounts but omitted the processing timelines the reference includes. The
remaining five are genuine:

1. **A trap was answered instead of refused.** "How do I renew my Pakistani
   passport?" produced identity-card renewal steps. This is the most serious
   failure — the grounding guarantee is the point of the system.
2. **"How long does executive processing take for a new CNIC?"** answered "not
   specified" although retrieval returned the fee schedule, which states 6 days.
3. **A Roman Urdu regression.** "Naya CNIC banwane ki fees kitni hai?" replied
   that the fee is unknown, while the identical question in English answered
   correctly — so the failure is language-specific, not a retrieval gap.
4. **NICOP document list** returned renewal requirements instead of the form's
   document list, despite correct retrieval.
5. **Address change** produced vague, partly invented guidance.

The pattern is consistent: **retrieval now puts the right document in front of
the model, and the model does not always use it.** That points at the prompt and
the answer-construction step, not the knowledge base.

---

## 8. Known limitations

- **Generation accuracy is the bottleneck** (§7.3), especially grounding
  discipline on traps and consistency across languages.
- **Free-tier token quota** caps evaluation. A full 54-question graded run does
  not fit in one day's budget; the quota is per organisation, so extra keys do
  not help.
- **Office-location data** is Google-sourced, not official; concentrated in major
  cities; and the compiling script was lost, so it cannot be refreshed.
- **Fee data** is a hand-made transcription — accurate to the published schedule
  (verified figure by figure) but it will need manual updating when NADRA
  changes fees. Note one anomaly reproduced faithfully from the source: Zone B
  Smart NICOP Duplicate Executive is listed at USD 140 where every sibling row is
  USD 40.
- **Fine-grained form questions** ("what blood group options are on the NICOP
  form?") rank poorly — the answer sits inside a chunk listing ~40 form fields.
- **Near-duplicate PDFs** (§3.2) remain un-deduplicated.

## 9. Recommended next steps

1. Fix the trap-refusal failure and the Roman Urdu fee regression — both are
   prompt-level, both are now reproducible from the gold set.
2. Re-run the full 54-question graded evaluation once quota allows, and track the
   score over time.
3. Rewrite and commit the Google Places collection script so the location data is
   reproducible.
4. Spot-check the two remaining vision transcriptions against their source pages,
   given the fabricated fee block found in the third (§4.3).
5. Decide on the near-duplicate PDF pairs.

## 10. Repository layout

```
src/ingest.py            PDFs + text sidecars → cleaned chunks → vector store
src/kb.py                embeddings, vector store, hybrid retriever, intent boosts
src/rag_pipeline.py      language handling, prompt, Groq call, citations
src/transcribe_forms.py  one-time vision transcription of scanned forms
app.py, frontend/        FastAPI API + chat UI
evaluation/              gold set, evaluate.py, targeted check scripts
pdfs_data/               source corpus (see below)
```

**Data distribution.** The 27 official NADRA PDFs (~226 MB) are re-downloadable
and stay out of git. The non-reproducible files — the hand-written transcription
sidecars including the fee schedule, and the compiled office-locations PDF — are
tracked. A fresh clone can answer fee and location questions before the PDFs are
added; `python src/ingest.py` rebuilds the index.
