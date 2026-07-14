"""Shared knowledge-base config: paths, embedding model, vector store access.

Both ingest.py (writer) and rag_pipeline.py (reader) import from here so the
embedding model and collection settings can never drift apart.
"""

import json
import re
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "pdfs_data"
TRANSCRIBED_DIR = PDF_DIR / "transcribed"
VECTOR_DIR = PROJECT_ROOT / "data" / "vector_store"
DOCUMENTS_FILE = VECTOR_DIR / "documents.json"
EMBEDDINGS_FILE = VECTOR_DIR / "embeddings.npy"
MANIFEST_FILE = VECTOR_DIR / "manifest.json"

# Multilingual E5 maps English, Urdu, and code-switched queries into the same
# embedding space. FastEmbed runs its official ONNX export locally on CPU.
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DOCUMENT_PREFIX = "passage: "
QUERY_PREFIX = "query: "
EMBEDDING_DIMENSION = 384


def _register_embedding_model():
    """Register the official ONNX export not bundled in FastEmbed's catalog."""
    supported = {item["model"].lower() for item in TextEmbedding.list_supported_models()}
    if EMBEDDING_MODEL.lower() not in supported:
        TextEmbedding.add_custom_model(
            model=EMBEDDING_MODEL,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=EMBEDDING_MODEL),
            dim=EMBEDDING_DIMENSION,
            model_file="onnx/model.onnx",
            description="Multilingual E5 Small (94 languages)",
            license="mit",
            size_in_gb=0.5,
        )


class FastEmbedEmbeddings(Embeddings):
    """Minimal LangChain Embeddings adapter over fastembed's ONNX runtime."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        # Cache under data/ (default is the Temp folder, which Windows may wipe).
        _register_embedding_model()
        cache = PROJECT_ROOT / "data" / "fastembed_cache"
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache))

    def embed_documents(self, texts):
        passages = (DOCUMENT_PREFIX + text for text in texts)
        return [vec.tolist() for vec in self._model.embed(passages)]

    def embed_query(self, text):
        return next(self._model.embed([QUERY_PREFIX + text])).tolist()


_embeddings = None
_vectorstore = None


def get_embeddings() -> FastEmbedEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = FastEmbedEmbeddings()
    return _embeddings


class LocalVectorStore:
    """Small persisted cosine-similarity store for the local NADRA corpus."""

    def __init__(self, documents: list[Document], vectors: np.ndarray):
        self._documents = documents
        self._vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
        self._vectors = self._vectors / np.maximum(norms, 1e-12)

    def get(self, include=None):
        return {
            "documents": [doc.page_content for doc in self._documents],
            "metadatas": [doc.metadata for doc in self._documents],
        }

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        vector = np.asarray(get_embeddings().embed_query(query), dtype=np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        scores = self._vectors @ vector
        order = np.argsort(-scores)[:k]
        return [self._documents[int(index)] for index in order]


def build_vectorstore(documents: list[Document]) -> LocalVectorStore:
    global _vectorstore
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    texts = [doc.page_content for doc in documents]
    vectors = np.asarray(get_embeddings().embed_documents(texts), dtype=np.float32)
    np.save(EMBEDDINGS_FILE, vectors, allow_pickle=False)
    payload = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in documents
    ]
    DOCUMENTS_FILE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    MANIFEST_FILE.write_text(
        json.dumps({"embedding_model": EMBEDDING_MODEL, "dimension": vectors.shape[1]}),
        encoding="utf-8",
    )
    _vectorstore = LocalVectorStore(documents, vectors)
    return _vectorstore


def get_vectorstore() -> LocalVectorStore:
    global _vectorstore
    if _vectorstore is None:
        if not DOCUMENTS_FILE.exists() or not EMBEDDINGS_FILE.exists() or not MANIFEST_FILE.exists():
            raise FileNotFoundError(
                "Knowledge base not found. Run `python src/ingest.py` first."
            )
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        if manifest.get("embedding_model") != EMBEDDING_MODEL:
            raise RuntimeError(
                "Knowledge base uses a different embedding model. "
                "Run `python src/ingest.py` to rebuild it."
            )
        payload = json.loads(DOCUMENTS_FILE.read_text(encoding="utf-8"))
        documents = [
            Document(page_content=item["page_content"], metadata=item["metadata"])
            for item in payload
        ]
        vectors = np.load(EMBEDDINGS_FILE, allow_pickle=False)
        _vectorstore = LocalVectorStore(documents, vectors)
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
