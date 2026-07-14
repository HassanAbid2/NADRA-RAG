"""Shared knowledge-base config: paths, embedding model, vector store access.

Both ingest.py (writer) and rag_pipeline.py (reader) import from here so the
embedding model and collection settings can never drift apart.
"""

import re
from pathlib import Path

from fastembed import TextEmbedding
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "pdfs_data"
TRANSCRIBED_DIR = PDF_DIR / "transcribed"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "nadra_kb"

# Same all-MiniLM-L6-v2 model as before, but run through ONNX (fastembed)
# instead of PyTorch: identical vectors, ~4s import instead of ~100s.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class FastEmbedEmbeddings(Embeddings):
    """Minimal LangChain Embeddings adapter over fastembed's ONNX runtime."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        # Cache under data/ (default is the Temp folder, which Windows may wipe).
        cache = PROJECT_ROOT / "data" / "fastembed_cache"
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache))

    def embed_documents(self, texts):
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


_embeddings = None
_vectorstore = None


def get_embeddings() -> FastEmbedEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = FastEmbedEmbeddings()
    return _embeddings


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=get_embeddings(),
            collection_name=COLLECTION_NAME,
        )
    return _vectorstore


_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "do", "does", "i", "my", "you", "your", "what", "which", "how", "can",
    "with", "at", "by", "be", "it", "this", "that", "will", "shall", "if",
    "from", "as", "any", "all", "when", "where", "who", "there",
}

_SUFFIXES = ("ments", "ment", "tions", "tion", "ings", "ing", "ies", "es", "ed", "al", "s")


def _tokenize(text: str) -> list[str]:
    """Lowercase words minus stopwords, crudely stemmed so that query terms
    like "renew"/"documents" match document terms "renewal"/"document"."""
    tokens = []
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        if tok in _STOPWORDS:
            continue
        for suffix in _SUFFIXES:
            if tok.endswith(suffix) and len(tok) > len(suffix) + 2:
                tok = tok[: -len(suffix)]
                break
        tokens.append(tok)
    return tokens


class HybridRetriever:
    """Vector similarity + BM25 keyword search, fused with Reciprocal Rank Fusion.

    Vectors capture paraphrases ("get a new card" ~ "renewal"); BM25 captures
    exact terms the embedding model underweights (fees, form field names,
    service acronyms). RRF combines both rankings without score calibration.
    """

    def __init__(self):
        data = get_vectorstore().get(include=["documents", "metadatas"])
        self._docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(data["documents"], data["metadatas"])
        ]
        self._bm25 = BM25Okapi([_tokenize(d.page_content) for d in self._docs])

    def search(self, query: str, k: int = 5) -> list[Document]:
        fetch_k = max(4 * k, 20)

        vector_docs = get_vectorstore().similarity_search(query, k=fetch_k)
        bm25_scores = self._bm25.get_scores(_tokenize(query))
        bm25_order = sorted(range(len(self._docs)), key=lambda i: -bm25_scores[i])
        bm25_docs = [self._docs[i] for i in bm25_order[:fetch_k] if bm25_scores[i] > 0]

        fused: dict[str, tuple[float, Document]] = {}
        for ranking in (vector_docs, bm25_docs):
            for rank, doc in enumerate(ranking):
                key = doc.page_content[:200]
                score = fused.get(key, (0.0, doc))[0] + 1.0 / (60 + rank)
                fused[key] = (score, doc)
        ranked = sorted(fused.values(), key=lambda pair: -pair[0])
        return [doc for _, doc in ranked[:k]]


_retriever = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
