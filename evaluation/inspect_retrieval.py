"""Print ranked evidence for representative NADRA questions."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_pipeline import retrieve  # noqa: E402

QUESTIONS = [
    "CNIC kesay renew hoga?",
    "What documents are required to renew a CNIC?",
    "NICOP banwane ki fees aur process kya hai?",
    "How can I track my NADRA application?",
    "FRC ke liye kaun eligible hai?",
]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for question in QUESTIONS:
        print(f"\n=== {question} ===")
        for index, document in enumerate(retrieve(question, k=10), start=1):
            text = " ".join(document.page_content.split())
            print(
                f"{index}. {document.metadata.get('source')} "
                f"p{document.metadata.get('page')}: {text[:300]}"
            )


if __name__ == "__main__":
    main()
