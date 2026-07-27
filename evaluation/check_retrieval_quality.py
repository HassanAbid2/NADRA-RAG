"""Offline regression checks for intent-aware NADRA retrieval."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from langchain_core.documents import Document  # noqa: E402

from rag_pipeline import normalize_question, retrieve, sources_used_by_answer  # noqa: E402


CASES = [
    ("CNNIC kesay renew hoga?", "renewal-guidelines.pdf"),
    ("What documents are required to renew a CNIC?", "renewal-guidelines.pdf"),
    ("NICOP banwane ki fees aur process kya hai?", "new-nicop.pdf"),
    ("How can I track my NADRA application?", "application-tracking.pdf"),
    ("FRC ke liye kaun eligible hai?", "frc-guide-v2.pdf"),
]


def main():
    for question, expected_source in CASES:
        documents = retrieve(normalize_question(question), k=3)
        sources = [document.metadata.get("source") for document in documents]
        assert expected_source in sources, (
            f"{question!r}: expected {expected_source} in top 3, got {sources}"
        )
        print(f"PASS: {question} -> {sources}")

    citation_docs = [
        Document("", metadata={"source": "guide.pdf", "page": page})
        for page in (2, 3, 4, 8)
    ]
    citations = sources_used_by_answer(
        citation_docs,
        "Sources: [guide.pdf, pages 2-4]",
    )
    assert citations == [
        {"source": "guide.pdf", "page": 2},
        {"source": "guide.pdf", "page": 3},
        {"source": "guide.pdf", "page": 4},
    ]
    assert sources_used_by_answer(
        citation_docs,
        "Sources: guide.pdf",
    ) == [{"source": "guide.pdf", "page": None}]
    print("PASS: structured citation filtering")


if __name__ == "__main__":
    main()
