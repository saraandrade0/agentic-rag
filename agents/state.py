"""Graph state definition for the Agentic RAG pipeline."""

from typing import List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State passed between nodes in the LangGraph pipeline.

    Attributes:
        question: The user's original question.
        documents: Retrieved documents after hybrid search.
        generation: The LLM-generated answer.
        search_type: Whether the query needs retrieval or direct answer.
        relevance_scores: Grading results for each retrieved document.
        retry_count: Number of retrieval retries (to prevent infinite loops).
    """

    question: str
    documents: List[dict]
    generation: Optional[str]
    search_type: Optional[str]
    relevance_scores: List[str]
    retry_count: int
