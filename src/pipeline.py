"""
pipeline.py — Teaching assistant pipeline.

Wraps the RAG pipeline with mode-aware prompting.
Each mode gets its own system prompt and output handling.
"""

from __future__ import annotations
import json
import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # no-op on Streamlit Cloud, loads .env locally

from src.modes import MODES, Mode
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.ingestion.embedder import Embedder


def _get_secret(key: str) -> str | None:
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key)


@st.cache_resource(show_spinner="Loading pipeline...")
def load_pipeline() -> "TeachingPipeline":
    for key in ("OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"):
        val = _get_secret(key)
        if val:
            os.environ[key] = val
    return TeachingPipeline()


class TeachingPipeline:
    COLLECTION = "d2l-book"
    MODEL = "gpt-4o-mini"

    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = VectorStore(self.COLLECTION)
        self.bm25_store = BM25Store()
        self.retriever = HybridRetriever(self.vector_store, self.bm25_store, self.embedder)
        self.reranker = Reranker()
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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
        Run one turn of the teaching assistant.

        Returns:
          {
            "raw": str,           # full LLM response text
            "parsed": dict|None,  # parsed JSON for quiz/flashcard modes
            "sources": list[dict]
          }
        """
        mode: Mode = MODES[mode_key]

        # 1. Retrieve relevant passages
        chunks = self.retriever.retrieve(user_message, top_k=top_k * 2)
        if len(chunks) > top_k:
            chunks = self.reranker.rerank(user_message, chunks, top_n=top_k)

        # 2. Build context block
        context = self._build_context(chunks)

        # 3. Build messages list (supports multi-turn for Socratic mode)
        messages = self._build_messages(mode, context, user_message, history)

        # 4. Call LLM
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content.strip()

        # 5. Try JSON parse for structured modes
        parsed = None
        if mode_key in ("quiz", "flashcards"):
            parsed = self._safe_parse_json(raw)

        sources = [
            {
                "source": c.metadata.get("source", "d2l-en.pdf"),
                "page": c.metadata.get("page", ""),
                "text": c.text[:300],
                "score": round(c.score, 3),
            }
            for c in chunks
        ]

        return {"raw": raw, "parsed": parsed, "sources": sources}

    # ------------------------------------------------------------------ #
    #  QUIZ ANSWER CHECKING                                               #
    # ------------------------------------------------------------------ #

    def check_answers(self, quiz: dict, user_answers: dict) -> dict:
        """
        Grade a completed quiz.
        quiz: the parsed quiz JSON from run()
        user_answers: {question_id: user_answer_string}
        Returns: {score, total, feedback: [{id, correct, explanation}]}
        """
        questions = quiz.get("questions", [])
        feedback = []
        score = 0

        for q in questions:
            qid = str(q["id"])
            user_ans = user_answers.get(qid, "").strip().lower()
            correct_ans = str(q["answer"]).strip().lower()

            # For MCQ: match letter (A/B/C/D)
            if q["type"] == "mcq":
                correct = user_ans and user_ans[0] == correct_ans[0]
            elif q["type"] == "true_false":
                correct = user_ans in ("true", "t") and correct_ans in ("true", "t") or \
                          user_ans in ("false", "f") and correct_ans in ("false", "f")
            else:
                # Short answer: use LLM to judge
                correct = self._llm_judge_answer(q["question"], correct_ans, user_ans)

            if correct:
                score += 1
            feedback.append({
                "id": q["id"],
                "question": q["question"],
                "user_answer": user_answers.get(qid, "(no answer)"),
                "correct_answer": q["answer"],
                "is_correct": correct,
                "explanation": q.get("explanation", ""),
            })

        return {"score": score, "total": len(questions), "feedback": feedback}

    # ------------------------------------------------------------------ #
    #  HELPERS                                                             #
    # ------------------------------------------------------------------ #

    def _build_messages(
        self,
        mode: Mode,
        context: str,
        user_message: str,
        history: list[dict],
    ) -> list[dict]:
        system = mode.system_prompt + f"\n\n<source_passages>\n{context}\n</source_passages>"

        messages = [{"role": "system", "content": system}]

        # Include conversation history for Socratic mode (multi-turn)
        for turn in history[-6:]:  # last 3 exchanges max
            messages.append({"role": turn["role"], "content": turn["content"]})

        messages.append({"role": "user", "content": user_message})
        return messages

    def _build_context(self, chunks) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            src = c.metadata.get("source", "d2l-en.pdf")
            page = c.metadata.get("page", "")
            header = f"[{i}] {src}" + (f", p.{page}" if page else "")
            parts.append(f"{header}\n{c.text}")
        return "\n\n---\n\n".join(parts)

    def _safe_parse_json(self, text: str) -> dict | None:
        # Strip markdown fences if present
        clean = text.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
        if clean.endswith("```"):
            clean = "\n".join(clean.split("\n")[:-1])
        try:
            return json.loads(clean.strip())
        except Exception:
            return None

    def _llm_judge_answer(self, question: str, correct: str, user: str) -> bool:
        """Use GPT to judge short-answer correctness."""
        prompt = f"""Question: {question}
Correct answer: {correct}
Student answer: {user}

Is the student's answer essentially correct? Reply with only "yes" or "no"."""
        r = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0,
        )
        return "yes" in r.choices[0].message.content.lower()
