"""Evaluate the NADRA RAG system: retrieval hit rate, refusal accuracy, and
(LLM-judged) answer correctness.

Usage:
    python evaluation/evaluate.py
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from langchain_groq import ChatGroq  # noqa: E402

from rag_pipeline import answer_question, retrieve  # noqa: E402

REFUSAL_MARKER = "I don't have verified NADRA information"
JUDGE_MODEL = "llama-3.3-70b-versatile"

JUDGE_PROMPT = """You are grading an FAQ system's answer against a reference answer.

Question: {question}
Reference answer: {reference}
System answer: {answer}

Does the system answer convey the same key facts as the reference (documents, fees,
timelines, eligibility)? Minor wording differences are fine; missing or wrong facts
are not. Reply with exactly one word: CORRECT or INCORRECT."""


def main():
    data = json.loads((PROJECT_ROOT / "evaluation" / "test_questions.json").read_text())
    questions = data["questions"]
    judge = ChatGroq(model=JUDGE_MODEL, temperature=0)

    retrieval_hits = 0
    retrieval_total = 0
    refusal_correct = 0
    refusal_total = 0
    answer_correct = 0
    answer_total = 0
    failures = []

    for q in questions:
        result = answer_question(q["question"])
        answer = result["answer"]
        refused = REFUSAL_MARKER.lower() in answer.lower()

        if q["should_refuse"]:
            refusal_total += 1
            if refused:
                refusal_correct += 1
            else:
                failures.append((q["id"], "answered an out-of-scope question", answer[:120]))
        else:
            # Retrieval: is the expected source among the top-k chunks?
            retrieval_total += 1
            docs = retrieve(q["question"])
            sources = {d.metadata.get("source") for d in docs}
            if q["expected_source"] in sources:
                retrieval_hits += 1
            else:
                failures.append((q["id"], f"expected {q['expected_source']}, got {sorted(sources)}", ""))

            # Generation: LLM-as-judge vs reference answer (skip TODO stubs)
            ref = q.get("reference_answer") or ""
            if ref and not ref.startswith("TODO"):
                answer_total += 1
                verdict = judge.invoke(JUDGE_PROMPT.format(
                    question=q["question"], reference=ref, answer=answer,
                )).content.strip().upper()
                if "CORRECT" in verdict and "INCORRECT" not in verdict:
                    answer_correct += 1
                else:
                    failures.append((q["id"], "judged incorrect", answer[:120]))

        time.sleep(1)  # stay well inside Groq free-tier rate limits
        print(f"  [{q['id']:>3}] done")

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    if retrieval_total:
        print(f"Retrieval hit rate : {retrieval_hits}/{retrieval_total} "
              f"({100 * retrieval_hits / retrieval_total:.0f}%)")
    if refusal_total:
        print(f"Refusal accuracy   : {refusal_correct}/{refusal_total} "
              f"({100 * refusal_correct / refusal_total:.0f}%)")
    if answer_total:
        print(f"Answer correctness : {answer_correct}/{answer_total} "
              f"({100 * answer_correct / answer_total:.0f}%)")
    else:
        print("Answer correctness : (no graded references yet — fill in reference_answer fields)")

    if failures:
        print("\nFailures:")
        for fid, reason, detail in failures:
            print(f"  [{fid}] {reason}" + (f" | {detail}" if detail else ""))


if __name__ == "__main__":
    main()
