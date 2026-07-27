"""Fast, offline regression checks for response-language selection.

Usage:
    python evaluation/check_language_policy.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_pipeline import (  # noqa: E402
    classify_question_language,
    clean_roman_urdu_vocabulary,
    contains_devanagari,
    contains_hindi_leaning_roman_terms,
    contains_urdu_script,
    greeting_response,
    is_language_followup,
    language_instruction,
    normalize_question,
)


def main():
    cases = {
        "What documents are required for a CNIC?": "english",
        "NICOP banwane ki fees kya hai?": "roman_urdu",
        "FRC ke liye kaun eligible hai?": "roman_urdu",
        "CNNIC kesay renew hoga?": "roman_urdu",
        "reply in urdu": "urdu",
        "reply in roman urdu": "roman_urdu",
        "شناختی کارڈ کی تجدید کیسے کروائیں؟": "urdu",
        "مجھے کون سے کاغذات درکار ہیں؟": "urdu",
    }

    for question, expected in cases.items():
        actual = classify_question_language(question)
        assert actual == expected, f"{question!r}: expected {expected}, got {actual}"

    assert contains_devanagari("यह हिंदी में है")
    assert contains_urdu_script("یہ اردو میں ہے")
    assert not contains_urdu_script("Yeh Roman Urdu mein hai")
    assert not contains_devanagari("یہ اردو میں ہے")
    assert not contains_devanagari("Yeh Roman Urdu mein hai")
    assert contains_hindi_leaning_roman_terms("Kripya sabhi jaankari dein")
    assert not contains_hindi_leaning_roman_terms(
        "Barah-e-karam tamam maloomat dein"
    )
    cleaned = clean_roman_urdu_vocabulary("Kripya sabhi jaankari dein")
    assert cleaned == "barah-e-karam tamam maloomat dein"

    assert "Devanagari" in language_instruction("NICOP ki fees kya hai?")
    assert "Pakistani Urdu" in language_instruction("یہ کیسے ہوگا؟")
    assert normalize_question("CNNIC kesay renew hoga?") == "CNIC kaise renew hoga?"
    assert is_language_followup("reply in urdu")
    assert is_language_followup("answer in Roman Urdu")
    assert "Wa Alaikum Assalam" in greeting_response("aoa")

    print(f"Language policy checks passed: {len(cases)} classification cases.")


if __name__ == "__main__":
    main()
