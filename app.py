"""
app.py — AI Teaching Assistant for Dive into Deep Learning
Run: streamlit run app.py
"""

import json
import os
import streamlit as st

# ── Pull secrets into env ─────────────────────────────────────────────────────
for key in ("OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"):
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

from src.pipeline import load_pipeline   # noqa: E402
from src.modes import MODES              # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="D2L Teaching Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 { font-family: 'DM Serif Display', serif; font-weight: 400; }

.mode-card {
    padding: 12px 16px;
    border-radius: 10px;
    border: 1px solid rgba(0,0,0,0.08);
    margin-bottom: 6px;
    cursor: pointer;
    transition: all 0.15s;
    background: white;
}
.mode-card:hover { border-color: #534AB7; }
.mode-card.selected { border-color: #534AB7; background: #f5f4ff; }

.source-chip {
    display: inline-block;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 20px;
    background: #f1f0fe;
    color: #534AB7;
    margin: 2px;
}

.quiz-q {
    background: #fafafa;
    border-left: 3px solid #534AB7;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 10px 0;
}
.difficulty-easy   { color: #0F6E56; font-weight: 500; font-size: 12px; }
.difficulty-medium { color: #BA7517; font-weight: 500; font-size: 12px; }
.difficulty-hard   { color: #993C1D; font-weight: 500; font-size: 12px; }

.flashcard {
    background: white;
    border: 1px solid rgba(0,0,0,0.1);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 8px 0;
    cursor: pointer;
    transition: box-shadow 0.15s;
}
.flashcard:hover { box-shadow: 0 2px 12px rgba(83,74,183,0.15); }
.card-front { font-weight: 500; font-size: 15px; margin-bottom: 6px; }
.card-back  { color: #555; font-size: 13px; line-height: 1.6; }
.card-cat   { font-size: 11px; padding: 2px 8px; border-radius: 20px; margin-top: 8px; display: inline-block; }
.cat-definition  { background: #e1f5ee; color: #0F6E56; }
.cat-formula     { background: #f5f4ff; color: #534AB7; }
.cat-intuition   { background: #faeeda; color: #BA7517; }
.cat-distinction { background: #faece7; color: #993C1D; }

.score-banner {
    padding: 16px 20px;
    border-radius: 12px;
    margin: 12px 0;
    text-align: center;
}
.score-great { background: #e1f5ee; border: 1px solid #9FE1CB; color: #0F6E56; }
.score-ok    { background: #faeeda; border: 1px solid #FAC775; color: #BA7517; }
.score-poor  { background: #faece7; border: 1px solid #F5C4B3; color: #993C1D; }

.correct-answer   { color: #0F6E56; font-weight: 500; }
.incorrect-answer { color: #993C1D; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "mode" not in st.session_state:
    st.session_state.mode = "explain"
if "history" not in st.session_state:
    st.session_state.history = []
if "quiz_state" not in st.session_state:
    st.session_state.quiz_state = None   # {quiz, answers, graded}
if "flashcard_deck" not in st.session_state:
    st.session_state.flashcard_deck = None
if "show_card_backs" not in st.session_state:
    st.session_state.show_card_backs = set()

pipeline = load_pipeline()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 D2L Teaching Assistant")
    st.caption("*Dive into Deep Learning* — all chapters")
    st.divider()

    st.markdown("**Choose a mode**")
    mode_icons = {"explain": "💡", "quiz": "✏️", "summarise": "📋",
                  "flashcards": "🗂️", "socratic": "🤔"}

    for key, mode in MODES.items():
        selected = st.session_state.mode == key
        if st.button(
            f"{mode_icons[key]} **{mode.name}**",
            key=f"mode_{key}",
            use_container_width=True,
            type="primary" if selected else "secondary",
        ):
            st.session_state.mode = key
            st.session_state.history = []
            st.session_state.quiz_state = None
            st.session_state.flashcard_deck = None
            st.session_state.show_card_backs = set()
            st.rerun()

    st.divider()
    top_k = st.slider("Retrieved chunks", 3, 10, 6)
    st.divider()

    # Example queries for current mode
    st.markdown("**Try asking:**")
    current_mode = MODES[st.session_state.mode]
    for q in current_mode.example_queries[:4]:
        if st.button(q, use_container_width=True, key=f"eg_{q[:20]}"):
            st.session_state["prefill"] = q
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.quiz_state = None
        st.session_state.flashcard_deck = None
        st.session_state.show_card_backs = set()
        st.rerun()

# ── Main content ──────────────────────────────────────────────────────────────
mode_key = st.session_state.mode
mode = MODES[mode_key]

col1, col2 = st.columns([6, 1])
with col1:
    st.markdown(f"## {mode_icons[mode_key]} {mode.name}")
    st.caption(mode.format_hint)

st.divider()

# ── Render conversation history ───────────────────────────────────────────────
for turn in st.session_state.history:
    if turn["role"] == "user":
        with st.chat_message("user"):
            st.markdown(turn["content"])
    else:
        with st.chat_message("assistant", avatar="🎓"):
            _render_mode = turn.get("mode", "explain")
            _parsed = turn.get("parsed")
            _raw = turn["content"]

            if _render_mode == "quiz" and _parsed:
                _render_quiz(turn)
            elif _render_mode == "flashcards" and _parsed:
                _render_flashcards(_parsed)
            else:
                st.markdown(_raw)

            if turn.get("sources"):
                with st.expander(f"📎 {len(turn['sources'])} source passages"):
                    for s in turn["sources"]:
                        pg = f", p.{s['page']}" if s.get("page") else ""
                        st.markdown(f"`{s['source']}{pg}` · score `{s['score']}`")
                        st.caption(s["text"] + "...")
                        st.divider()


def _render_quiz(turn: dict):
    """Render quiz questions with answer inputs and grading."""
    parsed = turn["parsed"]
    qs = parsed.get("questions", [])
    quiz_id = turn.get("quiz_id", "q0")

    st.markdown(f"**Quiz: {parsed.get('topic', 'Deep Learning')}** — {len(qs)} questions")

    if st.session_state.quiz_state and st.session_state.quiz_state.get("graded"):
        # Show graded results
        result = st.session_state.quiz_state["result"]
        pct = result["score"] / result["total"]
        cls = "score-great" if pct >= 0.8 else ("score-ok" if pct >= 0.5 else "score-poor")
        emoji = "🎉" if pct >= 0.8 else ("👍" if pct >= 0.5 else "📚")
        st.markdown(
            f'<div class="score-banner {cls}">'
            f'{emoji} <strong>{result["score"]}/{result["total"]}</strong> '
            f'({int(pct*100)}%)</div>',
            unsafe_allow_html=True,
        )
        for fb in result["feedback"]:
            icon = "✅" if fb["is_correct"] else "❌"
            with st.expander(f'{icon} Q{fb["id"]}: {fb["question"][:60]}...'):
                if not fb["is_correct"]:
                    st.markdown(f'Your answer: <span class="incorrect-answer">{fb["user_answer"]}</span>', unsafe_allow_html=True)
                    st.markdown(f'Correct answer: <span class="correct-answer">{fb["correct_answer"]}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="correct-answer">✓ {fb["correct_answer"]}</span>', unsafe_allow_html=True)
                st.info(fb["explanation"])
    else:
        # Show questions with input fields
        answers = {}
        for q in qs:
            diff_class = f"difficulty-{q.get('difficulty','medium')}"
            st.markdown(
                f'<div class="quiz-q">'
                f'<span class="{diff_class}">{q.get("difficulty","").upper()}</span> '
                f'· {q.get("type","").replace("_"," ").title()}<br>'
                f'<strong>Q{q["id"]}. {q["question"]}</strong></div>',
                unsafe_allow_html=True,
            )
            if q["type"] == "mcq" and q.get("options"):
                ans = st.radio(
                    f"Select answer for Q{q['id']}",
                    options=[opt[0] for opt in q["options"]],
                    format_func=lambda x, opts=q["options"]: next(o for o in opts if o.startswith(x)),
                    key=f"mcq_{quiz_id}_{q['id']}",
                    label_visibility="collapsed",
                )
                answers[str(q["id"])] = ans
            elif q["type"] == "true_false":
                ans = st.radio(
                    f"Q{q['id']}",
                    ["True", "False"],
                    key=f"tf_{quiz_id}_{q['id']}",
                    label_visibility="collapsed",
                    horizontal=True,
                )
                answers[str(q["id"])] = ans
            else:
                ans = st.text_area(
                    f"Your answer for Q{q['id']}",
                    key=f"sa_{quiz_id}_{q['id']}",
                    height=80,
                    label_visibility="collapsed",
                    placeholder="Type your answer...",
                )
                answers[str(q["id"])] = ans

        if st.button("✅ Check my answers", type="primary", key=f"grade_{quiz_id}"):
            result = pipeline.check_answers(parsed, answers)
            st.session_state.quiz_state = {"graded": True, "result": result}
            st.rerun()


def _render_flashcards(parsed: dict):
    """Render flashcard deck with flip-to-reveal."""
    cards = parsed.get("cards", [])
    st.markdown(f"**{parsed.get('topic', 'Flashcards')}** — {len(cards)} cards")
    st.caption("Click a card to reveal the answer.")

    for card in cards:
        cid = str(card["id"])
        revealed = cid in st.session_state.show_card_backs
        cat = card.get("category", "definition")

        col_a, col_b = st.columns([10, 1])
        with col_a:
            st.markdown(
                f'<div class="flashcard">'
                f'<div class="card-front">🃏 {card["front"]}</div>'
                + (f'<div class="card-back">→ {card["back"]}</div>' if revealed else "")
                + f'<span class="card-cat cat-{cat}">{cat}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_b:
            if st.button("👁" if not revealed else "🙈", key=f"flip_{cid}"):
                if revealed:
                    st.session_state.show_card_backs.discard(cid)
                else:
                    st.session_state.show_card_backs.add(cid)
                st.rerun()


# ── Chat input ────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", None)
prompt = st.chat_input(f"{mode.example_queries[0]}") or prefill

if prompt:
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.history.append({"role": "user", "content": prompt})

    # Get response
    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Thinking..."):
            # For Socratic: pass full history for multi-turn
            hist = st.session_state.history[:-1] if mode_key == "socratic" else []
            result = pipeline.run(mode_key, prompt, hist, top_k=top_k)

        raw = result["raw"]
        parsed = result["parsed"]

        import uuid
        quiz_id = str(uuid.uuid4())[:8]

        if mode_key == "quiz" and parsed:
            st.session_state.quiz_state = {"graded": False, "quiz": parsed, "answers": {}}
            turn = {"role": "assistant", "content": raw, "parsed": parsed,
                    "mode": mode_key, "sources": result["sources"], "quiz_id": quiz_id}
            _render_quiz(turn)
        elif mode_key == "flashcards" and parsed:
            st.session_state.flashcard_deck = parsed
            st.session_state.show_card_backs = set()
            turn = {"role": "assistant", "content": raw, "parsed": parsed,
                    "mode": mode_key, "sources": result["sources"]}
            _render_flashcards(parsed)
        else:
            st.markdown(raw)
            turn = {"role": "assistant", "content": raw, "parsed": parsed,
                    "mode": mode_key, "sources": result["sources"]}

        with st.expander(f"📎 {len(result['sources'])} source passages"):
            for s in result["sources"]:
                pg = f", p.{s['page']}" if s.get("page") else ""
                st.markdown(f"`{s['source']}{pg}` · score `{s['score']}`")
                st.caption(s["text"] + "...")
                st.divider()

    st.session_state.history.append(turn)
