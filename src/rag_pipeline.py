"""NADRA RAG pipeline: retrieve relevant chunks from ChromaDB, answer with Groq.

Requires GROQ_API_KEY in .env (free key from https://console.groq.com).
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

import kb

load_dotenv(kb.PROJECT_ROOT / ".env")

LLM_MODEL = "llama-3.3-70b-versatile"
TOP_K = 6

SYSTEM_PROMPT = """You are an official FAQ assistant for NADRA (National Database and \
Registration Authority, Pakistan). You answer citizen questions about NADRA services: \
CNIC, NICOP, POC, CRC (Child Registration Certificate), and FRC (Family Registration \
Certificate) — covering required documents, eligibility, fees, processing timelines, \
and procedures.

STRICT RULES:
1. Answer ONLY using the CONTEXT provided below. Never use outside knowledge.
2. If the context contains no relevant information at all, reply exactly: "I don't \
have verified NADRA information to answer that. Please contact the NADRA helpline \
(1777) or visit www.nadra.gov.pk." Do not guess.
2b. If the context answers the question only partially, give the part that IS \
covered and clearly say which part is not covered by official NADRA documents.
3. Answer in clear, plain English that an ordinary citizen can understand.
4. When the context gives fees, timelines, or document lists, state them precisely as \
written — do not round or paraphrase numbers.
5. End your answer with a "Sources:" line listing the document names and pages you used.

CONTEXT:
{context}"""

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(model=LLM_MODEL, temperature=0)
    return _llm


def warm_up():
    """Load the embedding model, vector store, and BM25 index up front."""
    kb.get_retriever().search("warm up", k=1)


def retrieve(question: str, k: int = TOP_K):
    """Return the top-k most relevant chunks (hybrid vector + keyword search)."""
    return kb.get_retriever().search(question, k=k)


def format_context(docs) -> str:
    parts = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[{src}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_question(question: str, k: int = TOP_K) -> dict:
    """Answer a citizen question, grounded in the NADRA knowledge base.

    Returns {"answer": str, "sources": [{"source": ..., "page": ...}, ...]}.
    """
    docs = retrieve(question, k=k)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{question}")]
    )
    chain = prompt | _get_llm()
    response = chain.invoke({"context": format_context(docs), "question": question})
    sources = [
        {"source": d.metadata.get("source"), "page": d.metadata.get("page")}
        for d in docs
    ]
    return {"answer": response.content, "sources": sources}
