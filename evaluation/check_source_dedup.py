"""One document cited on several pages must render as one source card."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_pipeline import sources_used_by_answer

docs = [
    SimpleNamespace(metadata={"source": "payment-v4.pdf", "page": p}) for p in (3, 5, 9)
] + [SimpleNamespace(metadata={"source": "frc-guide-v2.pdf", "page": 1})]

answer = "\n".join([
    "Fee details below.",
    "- payment-v4.pdf, page 3",
    "- payment-v4.pdf, page 5",
    "- payment-v4.pdf, page 9",
    "- frc-guide-v2.pdf, page 1",
])

result = sources_used_by_answer(docs, answer)
assert result == [
    {"source": "payment-v4.pdf", "page": "3, 5, 9"},
    {"source": "frc-guide-v2.pdf", "page": "1"},
], result
print("OK: one source card per document")
