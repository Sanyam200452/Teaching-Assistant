"""
pipeline.py — Teaching assistant pipeline using LangChain LCEL.

Each mode is a LangChain chain built with LCEL (LangChain Expression Language):
  retriever | prompt | llm | output_parser

Socratic mode uses ConversationBufferWindowMemory for multi-turn state.

pip install langchain langchain-openai
"""

from __future__ import annotations
import json
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

from src.modes import MODES, Mode
from src.ingestion.chunker import DocumentChunker, Chunk
from src.ingestion.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_retriever import HybridRetriever

load_dotenv()


def _get_secret(key: str) -> str | None:
    try:
        if hasattr(st, "secrets"):
            val = st.secrets.get(key)
            if val:
                return val
    except Exception:
        pass
    return os.environ.get(key)


@st.cache_resource(show_spinner="Loading pipeline...")
def load_pipeline() -> "TeachingPipeline":
    for key in ("OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "COHERE_API_KEY"):
        val = _get_secret(key)
        if val:
            os.environ[key] = val
    return TeachingPipeline()


class TeachingPipeline:
    MODEL      = "gpt-4o-mini"
    COLLECTION = "d2l-book"

    def __init__(self):
        self.embedder     = Embedder()
        self.vector_store = VectorStore(self.COLLECTION)
        self.bm25_store   = BM25Store()
        self.retriever    = HybridRetriever(
            self.vector_store,
            self.bm25_store,
            self.embedder,
        )
        self.llm = ChatOpenAI(
            model=self.MODEL,
            temperature=0.3,
            openai_api_key=os.environ["OPENAI_API_KEY"],
        )

    # ------------------------------------------------------------------ #
    #  MAIN ENTRY POINT                                                    #
    # ------------------------------------------------------------------ #

    def run(
        self,
        mode_key: str,
        user_message: str,
        history: list[dict],
        top_k: int = 6,
    ) -> dict:
        """
        Run one turn. Returns:
          {"raw": str, "parsed": dict|None, "sources": list[dict]}
        """
        mode: Mode = MODES[mode_key]

        # 1. Retrieve
        chunks = self.retriever.retrieve(user_message, top_k=top_k)
        context = self._build_context(chunks)

        # 2. Build and invoke the LCEL chain for this mode
        if mode_key == "socratic":
            raw = self._run_socratic(mode, user_message, context, history)
        else:
            raw = self._run_standard(mode, user_message, context)

        # 3. Parse JSON for structured modes
        parsed = None
        if mode_key in ("quiz", "flashcards"):
            parsed = self._safe_json(raw)

        sources = [
            {
                "source": c.metadata.get("source", "d2l-en.pdf"),
                "page":   c.metadata.get("page", ""),
                "text":   c.text[:300],
                "score":  round(c.score, 3),
            }
            for c in chunks
        ]

        return {"raw": raw, "parsed": parsed, "sources": sources}

    # ------------------------------------------------------------------ #
    #  LCEL CHAINS                                                         #
    # ------------------------------------------------------------------ #

    def _run_standard(self, mode: Mode, question: str, context: str) -> str:
        """
        Standard LCEL chain for non-Socratic modes:
          prompt | llm | StrOutputParser
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", mode.system_prompt + "\n\n<source_passages>\n{context}\n</source_passages>"),
            ("human", "{question}"),
        ])

        chain = prompt | self.llm | StrOutputParser()

        return chain.invoke({
            "context":  context,
            "question": question,
        })

    def _run_socratic(
        self,
        mode: Mode,
        question: str,
        context: str,
        history: list[dict],
    ) -> str:
        """
        Multi-turn LCEL chain for Socratic mode.
        Includes last 6 messages of history as MessagesPlaceholder.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", mode.system_prompt + "\n\n<source_passages>\n{context}\n</source_passages>"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        chain = prompt | self.llm | StrOutputParser()

        # Convert history dicts to LangChain message objects
        lc_history = []
        for turn in history[-6:]:
            if turn["role"] == "user":
                lc_history.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                lc_history.append(AIMessage(content=turn["content"]))

        return chain.invoke({
            "context":  context,
            "question": question,
            "history":  lc_history,
        })

    # ------------------------------------------------------------------ #
    #  QUIZ GRADING                                                        #
    # ------------------------------------------------------------------ #

    def check_answers(self, quiz: dict, user_answers: dict) -> dict:
        questions = quiz.get("questions", [])
        feedback  = []
        score     = 0

        for q in questions:
            qid      = str(q["id"])
            user_ans = user_answers.get(qid, "").strip().lower()
            corr_ans = str(q["answer"]).strip().lower()

            if q["type"] == "mcq":
                correct = bool(user_ans) and user_ans[0] == corr_ans[0]
            elif q["type"] == "true_false":
                correct = (
                    user_ans in ("true", "t")  and corr_ans in ("true", "t")  or
                    user_ans in ("false", "f") and corr_ans in ("false", "f")
                )
            else:
                correct = self._llm_judge(q["question"], corr_ans, user_ans)

            if correct:
                score += 1

            feedback.append({
                "id":           q["id"],
                "question":     q["question"],
                "user_answer":  user_answers.get(qid, "(no answer)"),
                "correct_answer": q["answer"],
                "is_correct":   correct,
                "explanation":  q.get("explanation", ""),
            })

        return {"score": score, "total": len(questions), "feedback": feedback}

    # ------------------------------------------------------------------ #
    #  HELPERS                                                             #
    # ------------------------------------------------------------------ #

    def _build_context(self, chunks: list[Chunk]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            src  = c.metadata.get("source", "d2l-en.pdf")
            page = c.metadata.get("page", "")
            hdr  = f"[{i}] {src}" + (f", p.{page}" if page else "")
            parts.append(f"{hdr}\n{c.text}")
        return "\n\n---\n\n".join(parts)

    def _safe_json(self, text: str) -> dict | None:
        clean = text.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
        if clean.endswith("```"):
            clean = "\n".join(clean.split("\n")[:-1])
        try:
            return json.loads(clean.strip())
        except Exception:
            return None

    def _llm_judge(self, question: str, correct: str, user: str) -> bool:
        """LLM-as-judge for short answer grading."""
        prompt = ChatPromptTemplate.from_messages([
            ("human", (
                f"Question: {question}\n"
                f"Correct answer: {correct}\n"
                f"Student answer: {user}\n\n"
                "Is the student's answer essentially correct? Reply with only 'yes' or 'no'."
            ))
        ])
        chain = prompt | ChatOpenAI(
            model=self.MODEL,
            temperature=0,
            max_tokens=5,
            openai_api_key=os.environ["OPENAI_API_KEY"],
        ) | StrOutputParser()

        return "yes" in chain.invoke({}).lower()
