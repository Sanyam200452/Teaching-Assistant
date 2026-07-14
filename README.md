# 🎓 AI Teaching Assistant for Deep Learning

A RAG-powered teaching assistant built on [*Dive into Deep Learning*](https://d2l.ai) (d2l.ai). Instead of a plain "chat with PDF" chatbot, it has **5 distinct teaching modes** — each with its own prompt, output schema, and UI — and one of them (Socratic tutoring) runs as a **LangGraph state machine** rather than a single prompt call.

**[🚀 Live Demo →](https://your-app.streamlit.app)**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C)
![LangGraph](https://img.shields.io/badge/LangGraph-state--machine-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)
![Qdrant](https://img.shields.io/badge/Vector_DB-Qdrant-DC244C)
![Cohere](https://img.shields.io/badge/Reranker-Cohere_v3-39594D)

---

## What it does

Pick a mode, ask about anything in the book:

| Mode | What it does |
|---|---|
| 💡 **Explain** | Breaks down a concept at beginner / intermediate / advanced level, with analogies and a worked example |
| ✏️ **Quiz me** | Generates MCQ, short-answer, and true/false questions from any chapter, with auto-grading and per-question feedback |
| 📋 **Summarise** | Structured chapter overview — core idea → key concepts → equations → connections → 3 things to remember |
| 🗂️ **Flashcards** | Extracts key terms as flippable cards tagged by type (definition / formula / intuition / distinction) |
| 🤔 **Socratic** | Guides you to the answer through questions — a **LangGraph** agent that tracks turns, detects understanding, and gives hints when you're stuck, instead of just following prompt instructions |

---

## Architecture

```
INDEXING (run once locally)               QUERYING (every message)
────────────────────────                  ─────────────────────────
d2l-en.pdf                                user message + mode
    │                                          │
    ▼                                          ├─────────────────┐
UnstructuredPDFLoader                          ▼                 ▼
(LangChain + unstructured.io,             Qdrant search      BM25 search
 structure-aware: code/tables/            (vector)           (keyword)
 equations kept intact)                        └────────┬────────┘
    │                                                   ▼
    ▼                                          EnsembleRetriever
RecursiveCharacterTextSplitter                 (weighted RRF fusion)
    │                                                   │
    ├──────────────────┐                                ▼
    ▼                  ▼                          CohereRerank
OpenAIEmbeddings   BM25Retriever                  (cross-encoder,
(text-embedding-   (rank-bm25,                     top-6 of 12)
 3-small)           pickled to disk)                    │
    │                                                    ▼
    ▼                                          mode routing
Qdrant Cloud                                             │
(persistent vector store)                    ┌───────────┴────────────┐
                                              ▼                        ▼
                                     4 modes: LCEL chain      Socratic: LangGraph
                                     prompt | llm | parser    evaluate → respond
                                                               │      → hint
                                                               │      → conclude
                                                               ▼
                                                    answer + source citations
```

---

## Key technical decisions

**Why LangChain's `UnstructuredPDFLoader` instead of a plain text splitter?**
The book is full of code blocks, LaTeX equations, and tables. A naive character splitter would cut a chunk mid-equation or mid-code-block. `UnstructuredPDFLoader(mode="elements")` classifies every element (`Title`, `NarrativeText`, `CodeSnippet`, `Table`) before any splitting happens, so structure is never destroyed by the chunker.

**Why hybrid search (`EnsembleRetriever`) instead of vector-only?**
Vector search is strong on paraphrased questions but weak on exact terms — model names, equation labels, acronyms like "Adam" or "ReLU". BM25 catches those. `EnsembleRetriever` fuses both ranked lists with weighted reciprocal rank fusion, giving consistently better recall than either alone with no score calibration required.

**Why Cohere Rerank on top of the ensemble?**
The retrievers above score query and chunk *separately* (bi-encoder-style) — fast but approximate. `CohereRerank` is a cross-encoder: it sees the (query, chunk) pair together and re-scores far more accurately. Pattern used here: retrieve 12 candidates broadly, rerank down to the best 6.

**Why LangGraph for Socratic mode specifically, and not the other 4 modes?**
Explain, Quiz, Summarise, and Flashcards are single-turn — one question in, one structured answer out. A linear LCEL chain (`prompt | llm | parser`) is the right tool there; adding a graph would be unnecessary complexity. Socratic mode is different: it needs to *decide* things — has the student understood, are they stuck, should the dialogue end. Those are branching decisions with real state (turn count, consecutive stuck replies). LangGraph makes that control flow explicit Python code with a hard turn cap, rather than hoping the LLM follows "please end the dialogue eventually" written into a prompt.

**Why structured JSON output for Quiz and Flashcards?**
Both modes instruct the LLM to return a strict JSON schema, which the app parses and renders as real interactive UI — radio buttons for MCQ, flip buttons for flashcards, per-question grading feedback — instead of plain text the user has to read through.

---

## Tech stack

| Component | Choice | Why |
|---|---|---|
| Orchestration | LangChain (LCEL) | Standard chain interface, composable |
| Multi-turn agent | LangGraph | Explicit state machine for Socratic mode |
| Document loading | `UnstructuredPDFLoader` | Structure-aware, keeps code/equations intact |
| Chunking | `RecursiveCharacterTextSplitter` | Paragraph → sentence → word fallback splitting |
| Embeddings | `text-embedding-3-small` | 1536 dims, cheap, strong quality |
| Vector DB | Qdrant Cloud | Free tier, in-memory local dev mode, no server setup |
| Keyword search | `BM25Retriever` (rank-bm25) | Lightweight, catches exact-term matches |
| Fusion | `EnsembleRetriever` | Weighted reciprocal rank fusion, no calibration needed |
| Reranking | `CohereRerank` v3 | Cross-encoder accuracy, generous free tier |
| LLM | GPT-4o-mini | Fast and cheap for dev; swap to GPT-4o for production |
| UI | Streamlit | Fast to build, deploys straight from GitHub |
| Deployment | Streamlit Cloud | Free tier, connects directly to the repo |

---

## Project structure

```
teaching-assistant/
├── app.py                          # Streamlit UI — sidebar, quiz/flashcard renderers, chat loop
├── run_indexing.py                 # One-time script: chunk, embed, upload the book
├── requirements.txt
├── .env.example                    # Copy to .env and fill in your keys
├── src/
│   ├── modes.py                    # 5 mode definitions — system prompts + output schemas
│   ├── pipeline.py                 # Orchestrator — retrieval + mode routing + quiz grading
│   ├── socratic_graph.py           # LangGraph state machine for Socratic mode
│   ├── ingestion/
│   │   ├── chunker.py              # UnstructuredPDFLoader + RecursiveCharacterTextSplitter
│   │   └── embedder.py             # OpenAIEmbeddings wrapper with query-level caching
│   └── retrieval/
│       ├── vector_store.py         # QdrantVectorStore wrapper
│       ├── bm25_store.py           # BM25Retriever with pickle persistence
│       └── hybrid_retriever.py     # EnsembleRetriever + CohereRerank pipeline
```

---

## Setup

### Prerequisites
- Python 3.10+
- [OpenAI API key](https://platform.openai.com/api-keys)
- [Qdrant Cloud](https://cloud.qdrant.io) cluster (free tier, 1GB)
- [Cohere API key](https://cohere.com) (free tier)
- The book PDF — [d2l.ai/d2l-en.pdf](https://d2l.ai/d2l-en.pdf)

### Local setup

```bash
git clone https://github.com/yourusername/teaching-assistant
cd teaching-assistant

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env          # then fill in your 4 keys
```

`.env` needs:
```
OPENAI_API_KEY=sk-...
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-key
COHERE_API_KEY=your-cohere-key
```

### Index the book (run once)

```bash
python run_indexing.py --pdf ./d2l-en.pdf
```

Takes a few minutes for the full book. Prints a metadata summary at the end (page numbers, code chunks) as a sanity check. Commit the resulting `bm25_index.pkl` — Streamlit Cloud needs it since it can't rebuild the BM25 index at startup.

```bash
git add bm25_index.pkl
git commit -m "add BM25 index"
```

### Run locally

```bash
streamlit run app.py
```

---

## Deployment (Streamlit Cloud)

1. Push the repo to GitHub (`bm25_index.pkl` committed, `.env` not)
2. [share.streamlit.io](https://share.streamlit.io) → New app → select your repo, main file `app.py`
3. **Settings → Secrets** → paste your 4 API keys
4. Deploy — live in a few minutes

---

## What I'd add next

- **Evaluation suite** — Hit Rate@5 and LLM-as-judge faithfulness scoring across a fixed test set
- **Chapter filtering** — restrict retrieval to a chosen chapter via Qdrant payload filters
- **Checkpointed LangGraph** — persist Socratic state with a proper LangGraph checkpointer instead of round-tripping `stuck_count` through Streamlit session state
- **Progress tracking** — remember quiz scores and chapters covered across sessions
- **Streaming responses** — token-by-token output instead of waiting for the full response

---

## License

MIT
