"""FastAPI app exposing the Agentic RAG pipeline.

Usage:
    uvicorn api.app:app --reload --port 8000

Endpoints:
    POST /query  — ask a question
    GET  /health — health check
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agents.graph import app as rag_graph

app = FastAPI(
    title="Agentic RAG API",
    description="LangGraph-powered RAG with hybrid search (semantic + BM25)",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    search_type: str | None
    num_documents: int


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Run a question through the Agentic RAG pipeline."""
    try:
        initial_state = {
            "question": request.question,
            "documents": [],
            "generation": None,
            "search_type": None,
            "relevance_scores": [],
            "retry_count": 0,
        }

        result = rag_graph.invoke(initial_state)

        sources = list({
            doc.get("metadata", {}).get("source", "unknown")
            for doc in result.get("documents", [])
        })

        return QueryResponse(
            question=request.question,
            answer=result.get("generation", "No answer generated."),
            sources=sources,
            search_type=result.get("search_type"),
            num_documents=len(result.get("documents", [])),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}
