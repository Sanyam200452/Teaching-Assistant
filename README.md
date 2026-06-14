#  AI Teaching Assistant for Deep Learning

A RAG-powered teaching assistant built on [Dive into Deep Learning (d2l.ai)](https://d2l.ai). Unlike a plain "chat with PDF" chatbot, it has **5 distinct teaching modes** — each with its own system prompt, output schema, and UI — making it a real learning tool rather than a tutorial clone.


---

## What it does

Ask it anything about the D2L textbook and pick a mode:

| Mode | What it does |
|---|---|
|  **Explain** | Breaks down any concept at beginner / intermediate / advanced level with analogies and examples |
|  **Quiz me** | Generates MCQ, short-answer, and true/false questions from any chapter — with auto-grading |
|  **Summarise** | Structured chapter overview: key concepts → equations → connections → 3 things to remember |
|  **Flashcards** | Extracts key terms as flippable cards tagged by type (definition, formula, intuition, distinction) |
|  **Socratic** | Guides you to the answer through questions — never gives it away directly |

---

## Architecture

```
INDEXING (run once locally)          QUERYING (every user message)
────────────────────────             ─────────────────────────────
d2l-en.pdf                           User message + mode
    │                                    │
    ▼                                    ├─────────────────────┐
unstructured.io                          ▼                     ▼
(structure-aware parsing)          Vector search           BM25 search
    │                              (Qdrant Cloud)          (keyword)
    ▼                                    └──────────┬──────────┘
chunk_by_title()                                    ▼
(respects section boundaries,                 RRF fusion
 keeps code blocks intact)                         │
    │                                              ▼
    ├──────────────────┐                      Reranker
    ▼                  ▼               (cross-encoder)
Embedder           BM25Store                       │
(text-embedding-   (rank-bm25,                     ▼
 3-small)           .pkl file)            Mode system prompt
    │                                    + retrieved context
    ▼                                              │
Qdrant Cloud                                       ▼
(persistent                                   GPT-4o-mini
 vector store)                                     │
                                                   ▼
                                        Answer + source citations
```

---

## Key technical decisions

**Why unstructured.io instead of a simple text splitter?**
The D2L book contains code blocks, LaTeX equations, and structured sections. A naive character splitter would cut mid-equation or split a PyTorch example across two chunks. `unstructured.io` classifies every element (NarrativeText, CodeSnippet, Formula, Table) before chunking, so code blocks and equations are always kept intact. `chunk_by_title()` starts a new chunk at every section heading, so chunks never span two topics.

**Why hybrid search (vector + BM25)?**
Vector search is great for paraphrased questions but misses exact terms — model names, equation labels, acronyms like "Adam" or "ReLU". BM25 catches those exact keyword matches. Combining both with **Reciprocal Rank Fusion (RRF)** consistently gives 5–15% better recall than either alone, with no score calibration needed.

**Why a reranker on top of hybrid?**
Bi-encoder embeddings score query and document separately — fast but approximate. A cross-encoder sees the (query, chunk) pair together, giving much more accurate relevance. Pattern: retrieve 12 candidates broadly, rerank to top 6 precisely.

**Why structured JSON output for Quiz and Flashcards?**
The LLM returns a strict JSON schema for these modes, which the app parses and renders as interactive UI elements — radio buttons for MCQ, flip buttons for flashcards, grading feedback per question. Falls back gracefully to plain text if JSON parsing fails.

---

## Tech stack

| Component | Choice | Why |
|---|---|---|
| Chunking | unstructured.io | Structure-aware, preserves code/equations |
| Embeddings | text-embedding-3-small | 1536 dims, cheap, excellent quality |
| Vector DB | Qdrant Cloud | Free tier, local dev mode, great filtering |
| Keyword search | rank-bm25 | Lightweight, no server needed |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | Free local fallback, no API key |
| LLM | GPT-4o-mini | Fast and cheap; swap to GPT-4o for production |
| UI | Streamlit | Fast to build, easy to deploy |
| Deployment | Streamlit Cloud | Free tier, connects to GitHub directly |

---

## Project structure

```
teaching-assistant/
├── app.py                          # Streamlit UI 
├── run_indexing.py                 # One-time script to index the book into Qdrant Cloud
├── requirements.txt
├── src/
│   ├── modes.py                    # 5 mode definitions: system prompts + output schemas
│   ├── pipeline.py                 # Core orchestrator — RAG + mode routing + quiz grading
│   ├── ingestion/
│   │   ├── chunker.py              # unstructured.io chunker (replaces naive text splitter)
│   │   └── embedder.py             # OpenAI embeddings with batching + LRU cache
│   └── retrieval/
│       ├── vector_store.py         # Qdrant Cloud wrapper
│       ├── bm25_store.py           # BM25 keyword index with pickle persistence
│       ├── hybrid_retriever.py     # RRF fusion of vector + BM25 results
│       └── reranker.py             # Cross-encoder reranker with Cohere fallback
```

---

### Index the book (run once)

```bash
python run_indexing.py --pdf ./d2l-en.pdf
```

---

## What I would add next

- **Evaluation suite** — automated Hit Rate@5 and LLM-as-judge faithfulness scoring across a 50-question test set
- **Chapter filter** — let users restrict retrieval to a specific chapter
- **Progress tracker** — remember which chapters you have quizzed on and your scores
- **Cohere reranker** — swap the local cross-encoder for Cohere Rerank v3 for better accuracy
- **HTML version support** — use the D2L HTML version for richer section metadata

---
