"""Shared knowledge-base config: paths, embedding model, vector store access.

Both ingest.py (writer) and rag_pipeline.py (reader) import from here so the
embedding model and collection settings can never drift apart.
"""

import json
import math
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


_QUERY_EXPANSIONS = (
    (r"\b(?:renew|renewal|tajdeed)\b", "renewal renew identity card application steps"),
    (r"\b(?:track|tracking|status)\b", "application tracking ID PIN status"),
    (r"\b(?:fee|fees|cost|price)\b", "fee fees payment cost"),
    (r"\b(?:banwana|banwane|apply|application)\b", "apply application process steps"),
    (r"\b(?:kaise|kesay|kese|how)\b", "how process steps"),
    (r"\b(?:eligible|eligibility|kaun|kon)\b", "eligible eligibility requirements"),
    # A "make/new CNIC or NICOP" query means fresh registration; steer BM25 and
    # vector search toward the fresh-registration policy pages, not reprint/renewal.
    (
        r"\b(?:make|new|fresh|first|create|banwa\w*)\b.*\b(?:cnic|nicop|id card|identity card)\b"
        r"|\b(?:cnic|nicop|id card|identity card)\b.*\b(?:make|new|fresh|first|create|banwa\w*)\b",
        "fresh new registration of citizens 18 years or above resident citizen "
        "application by applicant computerized birth certificate requirements "
        "with blood relative without blood relative system independent "
        "biometric verification affidavit attestation CNICF",
    ),
)


def _expand_query(query: str) -> str:
    """Add English retrieval terms for common Roman Urdu and intent phrases."""
    additions = [
        expansion
        for pattern, expansion in _QUERY_EXPANSIONS
        if re.search(pattern, query, re.IGNORECASE)
    ]
    return " ".join([query, *additions])


def _intent_source_boosts(query: str) -> dict[str, float]:
    """Prioritize the official guide that directly matches the user's intent."""
    lowered = query.lower()
    boosts: dict[str, float] = {}

    def add(source: str, score: float):
        boosts[source] = max(boosts.get(source, 0.0), score)

    if re.search(r"\b(?:track|tracking|status)\b", lowered):
        add("application-tracking.pdf", 0.045)
    if re.search(r"\bappointment|schedule|booking\b", lowered):
        add("appointment-scheduling.pdf", 0.045)
    if re.search(r"\b(?:renew|renewal|tajdeed)\b", lowered):
        add("renewal-guidelines.pdf", 0.045)
        add("registration-policy-6-0-1-english.pdf", 0.018)
        if re.search(r"\b(?:document|documents|required|requirement)\b", lowered):
            add("registration-policy-6-0-1-english.pdf", 0.035)
    if re.search(r"\bcnic\b", lowered) and re.search(
        r"\b(?:new|fresh|first|make|banwa|banwane|banwana|create|apply|application|get)\b",
        lowered,
    ):
        add("registration-policy-6-0-1-english.pdf", 0.045)
    if re.search(r"\b(?:reprint|lost|damaged)\b", lowered):
        add("reprint-guide.pdf", 0.045)
    if re.search(r"\bnicop\b", lowered):
        add("new-nicop.pdf", 0.035)
        add("nicop-complete-form-with-instruction.pdf", 0.025)
        add("registration-policy-6-0-1-english.pdf", 0.012)
        if re.search(r"\b(?:fee|fees|cost|price)\b", lowered):
            add("payment-v4.pdf", 0.025)
    if re.search(r"\bfrc\b|family registration certificate", lowered):
        add("frc-guide-v2.pdf", 0.04)
        add("registration-policy-6-0-1-english.pdf", 0.012)
        if re.search(r"\b(?:eligible|eligibility|kaun|kon)\b", lowered):
            add("registration-policy-6-0-1-english.pdf", 0.065)
    if re.search(r"\bpoc\b|pakistan origin card", lowered):
        add("new-smart-poc.pdf", 0.04)
        add("registration-policy-6-0-1-english.pdf", 0.006)
    if re.search(r"\bcrc\b|child registration certificate", lowered):
        add("new-crc-version-3-0.pdf", 0.04)
        add("registration-policy-6-0-1-english.pdf", 0.006)
    if re.search(r"\b(?:payment|pay|raast|easypaisa|jazzcash)\b", lowered):
        add("payment-v4.pdf", 0.045)
    # The authoritative itemized fee + timeline schedule lives in the dedicated
    # fee-structure sidecar, so route all fee/cost/timeline queries there.
    # It also carries the validity-period table, which lives nowhere else.
    if re.search(
        r"\b(?:fee|fees|cost|price|charges?|timeline|processing\s+time|"
        r"how\s+long|days|valid|validity|expire|expiry|kitni|kitna|kitne)\b",
        lowered,
    ):
        add("nadra-fee-structure.pdf", 0.06)
    # Office/branch/city-location queries — the only source with the address list
    # is NADRA_Office_Locations_Pakistan.pdf. Boost strongly so it wins even when
    # the query also mentions a service like "renew" or "cnic".
    # "address" alone is ambiguous: "change my address" is a modification request,
    # not a request for a branch address, so it only counts as a location signal
    # when no modification verb is present.
    place_words = re.search(
        r"\b(?:location|locations|office|offices|center|centre|centers|branch|"
        r"near\s*me|nearest|kahan|kahaan)\b",
        lowered,
    )
    bare_address = re.search(r"\baddress(?:es)?\b", lowered) and not re.search(
        r"\b(?:change|update|modify|correct|new|tabdeel|badal)\b", lowered
    )
    if place_words or bare_address:
        # Boost lookups compare against the lowercased source, so the key must be
        # lowercase even though the file itself is NADRA_Office_Locations_Pakistan.pdf.
        add("nadra_office_locations_pakistan.pdf", 0.12)
    if re.search(r"\bphoto|photograph\b", lowered):
        add("photgraph-guidelines-v4.pdf", 0.04)
        add("photo-guidelines.pdf", 0.04)
    if re.search(r"\bfingerprint|biometric\b", lowered):
        add("fingerprint-guidelines-v11.pdf", 0.04)
        add("fingerprint-guidelines.pdf", 0.04)
    if re.search(r"\bupload\b.*\bdocument|\bdocument\b.*\bupload", lowered):
        add("upload-document-guide.pdf", 0.045)
    return boosts


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
        expanded_query = _expand_query(query)

        vector_docs = get_vectorstore().similarity_search(expanded_query, k=fetch_k)
        bm25_scores = self._bm25.get_scores(_tokenize(expanded_query))
        bm25_order = sorted(range(len(self._docs)), key=lambda i: -bm25_scores[i])
        bm25_docs = [self._docs[i] for i in bm25_order[:fetch_k] if bm25_scores[i] > 0]

        fused: dict[str, tuple[float, Document]] = {}
        for ranking in (vector_docs, bm25_docs):
            for rank, doc in enumerate(ranking):
                key = doc.page_content[:200]
                score = fused.get(key, (0.0, doc))[0] + 1.0 / (60 + rank)
                fused[key] = (score, doc)

        source_boosts = _intent_source_boosts(query)
        focus_terms = [
            term
            for term in ("cnic", "nicop", "frc", "poc", "crc")
            if re.search(rf"\b{term}\b", query, re.IGNORECASE)
        ]

        # Ensure intent-matched documents participate even when a short or
        # Roman-Urdu query leaves their chunks just outside the global top-N.
        for source, boost in source_boosts.items():
            source_indices = [
                index
                for index, doc in enumerate(self._docs)
                if str(doc.metadata.get("source", "")).lower() == source
            ]
            source_indices.sort(key=lambda index: -bm25_scores[index])
            for source_rank, index in enumerate(source_indices[: max(3, k)]):
                doc = self._docs[index]
                key = doc.page_content[:200]
                base_score = fused.get(key, (0.0, doc))[0]
                lexical_bonus = 1.0 / (80 + source_rank)
                fused[key] = (base_score + lexical_bonus, doc)

        for key, (score, doc) in list(fused.items()):
            source = str(doc.metadata.get("source", "")).lower()
            source_boost = source_boosts.get(source, 0.0)
            if (
                source == "registration-policy-6-0-1-english.pdf"
                and focus_terms
                and not any(
                    re.search(rf"\b{term}\b", doc.page_content, re.IGNORECASE)
                    for term in focus_terms
                )
            ):
                source_boost *= 0.2
            score += source_boost
            fused[key] = (score, doc)

        ranked = sorted(fused.values(), key=lambda pair: -pair[0])
        max_per_source = max(3, math.ceil(k * 0.67))
        selected = []
        source_counts: dict[str, int] = {}
        selected_keys = set()
        for _, doc in ranked:
            source = str(doc.metadata.get("source", ""))
            if source_counts.get(source, 0) >= max_per_source:
                continue
            key = doc.page_content[:200]
            selected.append(doc)
            selected_keys.add(key)
            source_counts[source] = source_counts.get(source, 0) + 1
            if len(selected) == k:
                return selected

        for _, doc in ranked:
            key = doc.page_content[:200]
            if key not in selected_keys:
                selected.append(doc)
                selected_keys.add(key)
            if len(selected) == k:
                break
        return selected


_retriever = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
