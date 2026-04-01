# 🤖 Agentic RAG — Self-Correcting Retrieval with LangGraph

Agentic RAG pipeline that combines hybrid search (semantic + lexical), document grading, and hallucination detection using LangGraph for stateful orchestration.

Unlike a basic RAG pipeline that blindly retrieves and generates, this system uses an **agent loop** that can:
- **Route** queries (retrieval vs. direct answer)
- **Grade** retrieved documents for relevance
- **Retry** retrieval if no relevant documents are found
- **Detect hallucinations** and regenerate grounded answers

## 🏛️ Architecture

```
         ┌──────────┐
         │  Router  │ ← Classifies: needs retrieval?
         └─────┬────┘
          ┌────┴────┐
          │         │
          ▼         ▼
   ┌──────────┐  ┌────────────┐
   │ Retrieve │  │  Direct    │
   │ (Hybrid) │  │  Answer    │
   └─────┬────┘  └─────┬──────┘
         │              │
         ▼              │
   ┌──────────┐         │
   │  Grade   │         │
   │ Documents│         │
   └─────┬────┘         │
     ┌───┴───┐          │
     │       │          │
     ▼       ▼          │
  ┌──────┐  retry       │
  │ Gen  │   │          │
  └──┬───┘   │          │
     │       │          │
     ▼       │          │
  ┌──────────┐          │
  │Hallucin. │          │
  │  Check   │          │
  └────┬─────┘          │
   ┌───┴───┐            │
   │       │            │
   ▼       ▼            │
 done   regen.          │
   │       │            │
   ▼       ▼            ▼
  ┌──────────────────────────┐
  │           END            │
  └──────────────────────────┘
```

## 🛠️ Stack

| Component | Technology |
|-----------|-----------|
| Agent Orchestration | **LangGraph** |
| Semantic Search | **ChromaDB** + **sentence-transformers** |
| Lexical Search | **BM25** (rank-bm25) |
| Search Strategy | **Hybrid** (70% semantic + 30% lexical, RRF) |
| LLM | **OpenAI GPT-4o-mini** |
| Prompts | **LangChain** (ChatPromptTemplate) |
| API | **FastAPI** |
| Language | **Python 3.10+** |

## 📁 Structure

```
agentic-rag/
├── main.py                     # Interactive CLI
├── requirements.txt
├── .env.example
├── agents/
│   ├── graph.py                # LangGraph pipeline definition
│   ├── nodes.py                # Graph nodes (router, grader, generator, etc.)
│   └── state.py                # TypedDict state schema
├── tools/
│   └── retriever.py            # Hybrid search (ChromaDB + BM25)
├── scripts/
│   └── ingest.py               # PDF → chunks → ChromaDB
├── api/
│   └── app.py                  # FastAPI endpoint
└── data/
    ├── documents/              # Source PDFs (not versioned)
    └── chroma_db/              # Vector store (not versioned)
```

## 🚀 Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Set your OPENAI_API_KEY
```

### 3. Ingest documents
```bash
# Add PDFs to data/documents/, then:
python scripts/ingest.py --pdf-dir data/documents/

# Custom chunk size:
python scripts/ingest.py --pdf-dir data/documents/ --chunk-size 800 --chunk-overlap 100
```

### 4. Run

**Interactive CLI:**
```bash
python main.py
```

**API:**
```bash
uvicorn api.app:app --reload --port 8000
# POST http://localhost:8000/query  {"question": "..."}
```

## 🔍 Hybrid Search: How It Works

The retriever combines two strategies using **Reciprocal Rank Fusion (RRF)**:

1. **Semantic search (70% weight)** — Embeds the query with sentence-transformers and searches ChromaDB by cosine similarity. Captures meaning and intent.

2. **Lexical search (30% weight)** — BM25 over the full corpus. Catches exact term matches that embeddings might miss (acronyms, codes, proper nouns).

The RRF merge ensures that documents ranked highly by both methods get the strongest signal, while documents found by only one method still surface.

## 🧠 Agent Loop: Self-Correction

The LangGraph agent goes beyond simple retrieve-and-generate:

1. **Router** classifies the query — some questions don't need retrieval at all.
2. **Grader** evaluates each retrieved document for relevance, filtering noise.
3. **Retry logic** — if grading eliminates all documents, the agent retries retrieval (up to 2x) before falling back to a direct answer.
4. **Hallucination check** — after generation, verifies the answer is grounded in the source documents. If not, regenerates.

This makes the system more robust than a single-pass RAG pipeline, especially for ambiguous or complex queries.

## 📝 License

MIT
