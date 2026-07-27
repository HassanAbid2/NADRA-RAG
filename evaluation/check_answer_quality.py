"""Live answer-quality smoke tests using the configured Groq model.

Usage:
    python evaluation/check_answer_quality.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_pipeline import (  # noqa: E402
    answer_question,
    contains_devanagari,
    contains_urdu_script,
)


CASES = [
    {
        "question": "CNIC kesay renew hoga?",
        "required_terms": ("pakid", "id card", "renew"),
        "expected_source": "renewal-guidelines.pdf",
        "roman_urdu": True,
    },
    {
        "question": "What documents are required to renew a CNIC?",
        "required_terms": ("cnic", "document"),
        "expected_source": "registration-policy-6-0-1-english.pdf",
    },
    {
        "question": "NICOP banwane ki fees aur process kya hai?",
        "required_terms": ("nicop", "pakid", "fee"),
        "expected_source": "new-nicop.pdf",
        "roman_urdu": True,
    },
    {
        "question": "How can I track my NADRA application?",
        "required_terms": ("tracking id", "pin"),
        "expected_source": "application-tracking.pdf",
    },
    {
        "question": "FRC ke liye kaun eligible hai?",
        "required_terms": ("frc", "crc", "juvenile"),
        "expected_source": "registration-policy-6-0-1-english.pdf",
        "roman_urdu": True,
    },
]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for case in CASES:
        result = answer_question(case["question"])
        answer = result["answer"]
        answer_lower = answer.lower()
        sources = {source.get("source") for source in result["sources"]}

        missing = [
            term for term in case["required_terms"] if term not in answer_lower
        ]
        assert not missing, f"{case['question']!r}: missing terms {missing}\n{answer}"
        assert case["expected_source"] in sources, (
            f"{case['question']!r}: missing source {case['expected_source']}; "
            f"got {sorted(str(source) for source in sources)}"
        )
        assert not contains_devanagari(answer)
        if case.get("roman_urdu"):
            assert not contains_urdu_script(answer)

        print(f"\nPASS: {case['question']}")
        print(answer)
        print(f"Structured sources: {result['sources']}")


if __name__ == "__main__":
    main()
