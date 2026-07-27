"""Evaluate the NADRA RAG system: retrieval hit rate, refusal accuracy, and
(LLM-judged) answer correctness.

Retrieval runs entirely locally and costs nothing, so it covers the whole gold
set. Answer/refusal grading calls Groq, so --limit keeps a run inside the free
daily token budget.

Usage:
    python evaluation/evaluate.py --retrieval-only      # free, all questions
    python evaluation/evaluate.py --limit 20            # + LLM grading, capped
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from langchain_groq import ChatGroq  # noqa: E402

from rag_pipeline import answer_question, retrieve  # noqa: E402

REFUSAL_MARKER = "i don't have verified nadra information"
JUDGE_MODEL = "llama-3.3-70b-versatile"

JUDGE_PROMPT = """You are grading an FAQ system's answer against a reference answer.

Question: {question}
Reference answer: {reference}
System answer: {answer}

Does the system answer convey the same key facts as the reference (documents, fees,
timelines, eligibility)? Minor wording differences are fine; missing or wrong facts
are not. Reply with exactly one word: CORRECT or INCORRECT."""


def evaluate_retrieval(questions):
    """Local-only: is a document that actually holds the answer in the top-k?"""
    hits, total, misses = 0, 0, []
    for q in questions:
        if q["should_refuse"]:
            continue
        total += 1
        retrieved = [d.metadata.get("source") for d in retrieve(q["question"])]
        if set(q["expected_sources"]) & set(retrieved):
            hits += 1
        else:
            misses.append({
                "id": q["id"],
                "service": q["service"],
                "question": q["question"],
                "expected": q["expected_sources"],
                "retrieved": list(dict.fromkeys(retrieved)),
            })
    return hits, total, misses


def evaluate_answers(questions, limit):
    """Calls Groq: refusal accuracy on traps, LLM-judged correctness elsewhere."""
    judge = ChatGroq(model=JUDGE_MODEL, temperature=0)
    refusal_ok, refusal_total = 0, 0
    answer_ok, answer_total = 0, 0
    failures = []

    for q in questions[:limit]:
        try:
            answer = answer_question(q["question"])["answer"]
        except Exception as exc:  # rate limit, network, etc.
            failures.append({"id": q["id"], "reason": f"call failed: {exc}"[:160]})
            continue

        refused = REFUSAL_MARKER in answer.lower()
        if q["should_refuse"]:
            refusal_total += 1
            if refused:
                refusal_ok += 1
            else:
                failures.append({
                    "id": q["id"],
                    "reason": "answered an out-of-scope question",
                    "detail": answer[:160],
                })
        elif q["reference_answer"]:
            answer_total += 1
            verdict = judge.invoke(JUDGE_PROMPT.format(
                question=q["question"],
                reference=q["reference_answer"],
                answer=answer,
            )).content.strip().upper()
            if "INCORRECT" not in verdict and "CORRECT" in verdict:
                answer_ok += 1
            else:
                failures.append({
                    "id": q["id"],
                    "reason": "judged incorrect",
                    "detail": answer[:160],
                })
        time.sleep(1)  # stay inside Groq free-tier rate limits
        print(f"  [{q['id']:>3}] done")

    return {
        "refusal": (refusal_ok, refusal_total),
        "answer": (answer_ok, answer_total),
        "failures": failures,
    }


def pct(hits, total):
    return f"{hits}/{total} ({100 * hits / total:.0f}%)" if total else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true",
                    help="skip all Groq calls (free, no token spend)")
    ap.add_argument("--limit", type=int, default=54,
                    help="max questions to send to the LLM")
    ap.add_argument("--ids", default=None,
                    help="comma-separated question ids to grade (token budget)")
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args()

    data = json.loads((PROJECT_ROOT / "evaluation" / "test_questions.json")
                      .read_text(encoding="utf-8"))
    questions = data["questions"]

    print(f"Gold set: {len(questions)} questions "
          f"({sum(q['should_refuse'] for q in questions)} out-of-scope traps)\n")

    hits, total, misses = evaluate_retrieval(questions)
    results = {"retrieval": {"hits": hits, "total": total, "misses": misses}}

    print("=" * 52)
    print(f"Retrieval hit rate : {pct(hits, total)}")

    if not args.retrieval_only:
        graded_set = questions
        if args.ids:
            wanted = {int(i) for i in args.ids.split(",")}
            graded_set = [q for q in questions if q["id"] in wanted]
        graded = evaluate_answers(graded_set, args.limit)
        results["generation"] = graded
        print(f"Refusal accuracy   : {pct(*graded['refusal'])}")
        print(f"Answer correctness : {pct(*graded['answer'])}")
    print("=" * 52)

    if misses:
        print(f"\nRetrieval misses ({len(misses)}):")
        for m in misses:
            print(f"  [{m['id']}] {m['question']}")
            print(f"       expected any of {m['expected']}")
            print(f"       got {m['retrieved']}")

    if not args.retrieval_only and results["generation"]["failures"]:
        print("\nGeneration failures:")
        for f in results["generation"]["failures"]:
            print(f"  [{f['id']}] {f['reason']}"
                  + (f" | {f.get('detail', '')}" if f.get("detail") else ""))

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
