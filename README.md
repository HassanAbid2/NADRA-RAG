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
