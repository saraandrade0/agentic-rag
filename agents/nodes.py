"""LangGraph nodes for the Agentic RAG pipeline.

Each function is a node in the graph that takes the current state
and returns a partial state update.
"""

import json
from typing import Any, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from agents.state import AgentState
from tools.retriever import search

# --------------------------------------------------------------------------- #
# LLM setup
# --------------------------------------------------------------------------- #

import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    temperature=0,
)

# --------------------------------------------------------------------------- #
# Node: Router — decides if the query needs retrieval or direct answer
# --------------------------------------------------------------------------- #

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a query router. Analyze the user question and decide:\n"
        "- 'retrieve' → if the question requires searching a knowledge base\n"
        "- 'direct' → if you can answer from general knowledge (greetings, math, etc.)\n\n"
        "Respond with ONLY a JSON object: {{\"route\": \"retrieve\"}} or {{\"route\": \"direct\"}}"
    )),
    ("human", "{question}"),
])


def route_query(state: AgentState) -> Dict[str, Any]:
    """Classify if the question needs retrieval or can be answered directly."""
    chain = ROUTER_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"question": state["question"]})

    try:
        parsed = json.loads(result)
        search_type = parsed.get("route", "retrieve")
    except json.JSONDecodeError:
        search_type = "retrieve"

    return {"search_type": search_type}


# --------------------------------------------------------------------------- #
# Node: Retrieve — hybrid search (semantic + BM25)
# --------------------------------------------------------------------------- #


def retrieve(state: AgentState) -> Dict[str, Any]:
    """Run hybrid search and return documents."""
    documents = search(state["question"], k=5)
    return {
        "documents": documents,
        "retry_count": state.get("retry_count", 0),
    }


# --------------------------------------------------------------------------- #
# Node: Grade Documents — evaluate relevance of each retrieved doc
# --------------------------------------------------------------------------- #

GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a relevance grader. Given a user question and a retrieved document, "
        "determine if the document is relevant to answering the question.\n\n"
        "Respond with ONLY a JSON object: {{\"relevant\": \"yes\"}} or {{\"relevant\": \"no\"}}"
    )),
    ("human", "Question: {question}\n\nDocument:\n{document}"),
])


def grade_documents(state: AgentState) -> Dict[str, Any]:
    """Filter out irrelevant documents."""
    question = state["question"]
    documents = state["documents"]

    chain = GRADER_PROMPT | llm | StrOutputParser()

    relevant_docs = []
    scores = []

    for doc in documents:
        result = chain.invoke({
            "question": question,
            "document": doc.get("content", ""),
        })
        try:
            parsed = json.loads(result)
            grade = parsed.get("relevant", "no")
        except json.JSONDecodeError:
            grade = "no"

        scores.append(grade)
        if grade == "yes":
            relevant_docs.append(doc)

    return {
        "documents": relevant_docs,
        "relevance_scores": scores,
    }


# --------------------------------------------------------------------------- #
# Node: Generate — produce answer from relevant documents
# --------------------------------------------------------------------------- #

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful assistant that answers questions based on the provided context. "
        "Use ONLY the information from the context below. If the context doesn't contain "
        "enough information, say so clearly.\n\n"
        "Context:\n{context}"
    )),
    ("human", "{question}"),
])


def generate(state: AgentState) -> Dict[str, Any]:
    """Generate an answer from the relevant documents."""
    documents = state["documents"]
    question = state["question"]

    context = "\n\n---\n\n".join(
        f"[Source: {doc.get('metadata', {}).get('source', 'unknown')}]\n{doc.get('content', '')}"
        for doc in documents
    )

    chain = GENERATE_PROMPT | llm | StrOutputParser()
    generation = chain.invoke({"context": context, "question": question})

    return {"generation": generation}


# --------------------------------------------------------------------------- #
# Node: Direct Answer — answer without retrieval
# --------------------------------------------------------------------------- #

DIRECT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the question concisely."),
    ("human", "{question}"),
])


def direct_answer(state: AgentState) -> Dict[str, Any]:
    """Answer directly without retrieval (for simple queries)."""
    chain = DIRECT_PROMPT | llm | StrOutputParser()
    generation = chain.invoke({"question": state["question"]})
    return {"generation": generation}


# --------------------------------------------------------------------------- #
# Node: Hallucination Check — verify answer is grounded in documents
# --------------------------------------------------------------------------- #

HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a hallucination detector. Given a set of source documents and a generated "
        "answer, determine if the answer is grounded in the documents.\n\n"
        "Respond with ONLY a JSON object: {{\"grounded\": \"yes\"}} or {{\"grounded\": \"no\"}}"
    )),
    ("human", "Documents:\n{documents}\n\nGenerated answer:\n{generation}"),
])


def check_hallucination(state: AgentState) -> Dict[str, Any]:
    """Verify the generated answer is grounded in retrieved documents."""
    documents = state["documents"]
    generation = state["generation"]

    docs_text = "\n\n".join(doc.get("content", "") for doc in documents)
    chain = HALLUCINATION_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"documents": docs_text, "generation": generation})

    try:
        parsed = json.loads(result)
        grounded = parsed.get("grounded", "no")
    except json.JSONDecodeError:
        grounded = "no"

    # If not grounded and we haven't retried too many times, increment retry
    if grounded == "no" and state.get("retry_count", 0) < 2:
        return {"retry_count": state.get("retry_count", 0) + 1, "generation": None}

    return {}


# --------------------------------------------------------------------------- #
# Routing functions (conditional edges)
# --------------------------------------------------------------------------- #


def should_retrieve(state: AgentState) -> str:
    """Route based on query classification."""
    if state.get("search_type") == "direct":
        return "direct"
    return "retrieve"


def should_regenerate(state: AgentState) -> str:
    """Route based on hallucination check — regenerate if not grounded."""
    if state.get("generation") is None and state.get("retry_count", 0) < 2:
        return "regenerate"
    return "done"


def has_relevant_docs(state: AgentState) -> str:
    """Route based on whether grading found any relevant documents."""
    if not state.get("documents"):
        if state.get("retry_count", 0) < 2:
            return "retry"
        return "no_docs"
    return "generate"
