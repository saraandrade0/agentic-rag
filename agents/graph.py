"""LangGraph pipeline definition for Agentic RAG.

Builds a stateful graph with conditional routing:

    ┌─────────┐
    │  Router  │
    └────┬─────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌────────┐
│Retrieve│ │ Direct │
└───┬────┘ │ Answer │
    │      └───┬────┘
    ▼          │
┌────────┐    │
│ Grade  │    │
└───┬────┘    │
    │         │
  ┌─┴──┐     │
  │    │     │
  ▼    ▼     │
┌────┐ retry │
│Gen │  │    │
└─┬──┘  │    │
  │     │    │
  ▼     │    │
┌──────┐│    │
│Halluc││    │
│Check ││    │
└──┬───┘│    │
   │    │    │
   ▼    ▼    ▼
  ┌────────┐
  │  END   │
  └────────┘
"""

from langgraph.graph import END, StateGraph

from agents.nodes import (
    check_hallucination,
    direct_answer,
    generate,
    grade_documents,
    has_relevant_docs,
    retrieve,
    route_query,
    should_regenerate,
    should_retrieve,
)
from agents.state import AgentState


def build_graph() -> StateGraph:
    """Construct the Agentic RAG graph.

    Returns a compiled LangGraph that can be invoked with a question.
    """
    graph = StateGraph(AgentState)

    # --- Add nodes ---
    graph.add_node("router", route_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)
    graph.add_node("check_hallucination", check_hallucination)
    graph.add_node("direct_answer", direct_answer)

    # --- Entry point ---
    graph.set_entry_point("router")

    # --- Conditional: Router → Retrieve or Direct ---
    graph.add_conditional_edges(
        "router",
        should_retrieve,
        {
            "retrieve": "retrieve",
            "direct": "direct_answer",
        },
    )

    # --- Retrieve → Grade ---
    graph.add_edge("retrieve", "grade_documents")

    # --- Conditional: Grade → Generate, Retry, or No Docs ---
    graph.add_conditional_edges(
        "grade_documents",
        has_relevant_docs,
        {
            "generate": "generate",
            "retry": "retrieve",
            "no_docs": "direct_answer",
        },
    )

    # --- Generate → Hallucination Check ---
    graph.add_edge("generate", "check_hallucination")

    # --- Conditional: Hallucination → Done or Regenerate ---
    graph.add_conditional_edges(
        "check_hallucination",
        should_regenerate,
        {
            "regenerate": "generate",
            "done": END,
        },
    )

    # --- Direct Answer → End ---
    graph.add_edge("direct_answer", END)

    return graph.compile()


# Module-level compiled graph
app = build_graph()
