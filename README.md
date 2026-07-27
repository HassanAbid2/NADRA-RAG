# NADRA Guide

A document-grounded multilingual assistant for NADRA services. The retrieval,
embedding, and Groq answer pipeline remain in Python; the web experience uses a
FastAPI REST API and a responsive HTML/CSS/JavaScript frontend.

## Run the application

1. Create `.env` from `.env.example` and add your `GROQ_API_KEY`.
2. Install Python dependencies:

   ```powershell
   .\.venv\Scripts\python -m pip install -r requirements.txt
   ```

3. Start the complete application:

   ```powershell
   .\.venv\Scripts\python -m uvicorn app:app --reload
   ```

4. Open `http://127.0.0.1:8000`.

FastAPI serves the frontend source directly, so Node.js is not required for
the normal one-command run.

## Frontend development with Node.js

Run FastAPI on port 8000, then in another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api` requests to FastAPI.

Create an optimized production frontend with:

```powershell
cd frontend
npm run build
```

On its next start, FastAPI automatically serves `frontend/dist`.

## API

- `POST /api/chat` with `{"question": "..."}` returns the answer and source pages.
- `GET /api/health` returns service status.
- Interactive API documentation is available at `/docs`.

The knowledge-base workflow remains unchanged. To rebuild it after changing
the PDF collection, run:

```powershell
.\.venv\Scripts\python src\ingest.py
```

## Knowledge-base data

`pdfs_data/` holds the source corpus. Only part of it is in git:

- **Tracked** — `pdfs_data/transcribed/*.txt` (hand-written sidecars, including the
  authoritative `nadra-fee-structure.txt`) and `NADRA_Office_Locations_Pakistan.pdf`.
  These are not reproducible from the PDFs.
- **Not tracked** — the ~27 official NADRA PDFs (~226 MB), downloadable from
  [nadra.gov.pk](https://www.nadra.gov.pk/). Drop them into `pdfs_data/` and run
  `src\ingest.py` to build `data/vector_store`.

Ingestion reads both the PDFs and the sidecars, so a fresh clone answers fee and
office-location questions even before the PDFs are added.
