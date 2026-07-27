"""FastAPI web application for the NADRA Guide.

Development:
    uvicorn app:app --reload

Production:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DIST_DIR = FRONTEND_DIR / "dist"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from rag_pipeline import answer_question, warm_up  # noqa: E402

logger = logging.getLogger("nadra-guide")


class SourceReference(BaseModel):
    source: str | None = None
    page: str | int | None = None


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)
    sources: list[SourceReference] = Field(default_factory=list, max_length=10)


class ChatRequest(BaseModel):
    """Validated request body for a citizen question and recent conversation."""

    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatHistoryMessage] = Field(default_factory=list, max_length=12)


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceReference]


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Prepare the local retriever before accepting questions."""
    try:
        await asyncio.to_thread(warm_up)
        logger.info("NADRA knowledge base is ready.")
    except Exception:
        # Keep the web server available so /api/health can expose the issue and
        # the UI can display a useful error instead of failing to start.
        logger.exception("Knowledge-base warm-up failed.")
    yield


app = FastAPI(
    title="NADRA Guide API",
    description="Document-grounded FAQ assistant for NADRA identity services.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Please enter a question.")

    try:
        history = [message.model_dump() for message in payload.history]
        result = await asyncio.to_thread(
            answer_question,
            question,
            history=history,
        )
    except Exception as exc:
        logger.exception("Unable to answer question.")
        if getattr(exc, "status_code", None) == 429 or "rate limit" in str(exc).lower():
            detail = (
                "The free daily usage limit for the AI service has been reached. "
                "Please try again in a few minutes, or contact the NADRA helpline "
                "(1777) or visit www.nadra.gov.pk in the meantime."
            )
        else:
            detail = (
                "I couldn't complete that request. Please check the API "
                "connection and try again."
            )
        raise HTTPException(status_code=503, detail=detail) from exc

    return ChatResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
    )


# Prefer the optimized Vite build when present. The source frontend is also
# directly servable, which keeps local setup to a single Python command.
WEB_ROOT = DIST_DIR if (DIST_DIR / "index.html").exists() else FRONTEND_DIR
ASSETS_DIR = WEB_ROOT / "assets"
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/{path:path}", include_in_schema=False)
async def frontend_route(path: str) -> FileResponse:
    """Serve frontend files and retain an SPA-compatible fallback."""
    requested = (WEB_ROOT / path).resolve()
    try:
        requested.relative_to(WEB_ROOT.resolve())
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found.")

    if requested.is_file():
        return FileResponse(requested)
    return FileResponse(WEB_ROOT / "index.html")
