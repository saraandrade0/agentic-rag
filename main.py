"""Interactive CLI for the Agentic RAG pipeline.

Usage:
    python main.py
"""

from dotenv import load_dotenv

load_dotenv()

from agents.graph import app


def main():
    print("🤖 Agentic RAG — type 'quit' to exit\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break

        if not question:
            continue

        initial_state = {
            "question": question,
            "documents": [],
            "generation": None,
            "search_type": None,
            "relevance_scores": [],
            "retry_count": 0,
        }

        result = app.invoke(initial_state)

        print(f"\nAssistant: {result.get('generation', 'No answer generated.')}")

        if result.get("documents"):
            sources = {
                doc.get("metadata", {}).get("source", "unknown")
                for doc in result["documents"]
            }
            print(f"📚 Sources: {', '.join(sources)}")

        print(f"🔍 Route: {result.get('search_type', 'unknown')}\n")


if __name__ == "__main__":
    main()
