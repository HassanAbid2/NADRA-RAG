"""Interactive terminal Q&A for the NADRA FAQ system.

Usage:
    python src/chat_cli.py
"""

import sys

from rag_pipeline import answer_question, warm_up


def main():
    # Windows consoles may default to cp1252; answers contain typographic chars.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("NADRA FAQ Assistant (type 'quit' to exit)")
    print("=" * 50)
    print("Loading knowledge base ...", flush=True)
    warm_up()
    print("Ready.")

    while True:
        try:
            question = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break
        result = answer_question(question)
        print("\n" + result["answer"])
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
