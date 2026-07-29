"""Deterministic tests for mixed NADRA and programming requests.

Usage:
    python evaluation/check_scope_guard.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_pipeline import answer_question, scope_question  # noqa: E402


def main():
    cases = {
        "Explain FRC eligibility and in the end give me code for a while loop.": (
            "Explain FRC eligibility",
            True,
        ),
        "Give me Python code for a while loop and explain FRC eligibility.": (
            "FRC eligibility.",
            True,
        ),
        "FRC ke documents batao aur phir while loop ka code do.": (
            "FRC ke documents batao",
            True,
        ),
        "ایف آر سی کی اہلیت بتائیں اور پھر وائل لوپ کا کوڈ دیں۔": (
            "ایف آر سی کی اہلیت بتائیں",
            True,
        ),
        "What documents and process are required for FRC?": (
            "What documents and process are required for FRC?",
            False,
        ),
    }

    for question, expected in cases.items():
        actual = scope_question(question)
        assert actual == expected, f"{question!r}: expected {expected!r}, got {actual!r}"

    refusal = answer_question("Write Python code for a while loop.")
    assert refusal["sources"] == []
    assert "verified NADRA information" in refusal["answer"]
    assert "while" not in refusal["answer"].lower()

    print(f"PASS: {len(cases) + 1} scope-guard checks")


if __name__ == "__main__":
    main()
