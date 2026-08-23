"""
app.py — AI Teaching Assistant for Dive into Deep Learning.
Run: streamlit run app.py
"""

import os
import uuid
import streamlit as st


# Pull Streamlit secrets into env before any imports that need them
for key in ("OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "COHERE_API_KEY"):
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

from src.pipeline import load_pipeline   # noqa: E402
from src.modes import MODES              # noqa: E402

import json
 
@st.cache_data
def load_chapters() -> list[tuple[str, str]]:
    """Returns [(chapter_num, chapter_title), ...] sorted numerically."""
    try:
        with open("chapters.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
 
CHAPTERS = load_chapters()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="D2L Teaching Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; font-weight: 400; }

.quiz-q {
    background: #fafafa;
    border-left: 3px solid #534AB7;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 10px 0;
}
.diff-easy   { color: #0F6E56; font-weight: 500; font-size: 12px; }
.diff-medium { color: #BA7517; font-weight: 500; font-size: 12px; }
.diff-hard   { color: #993C1D; font-weight: 500; font-size: 12px; }

.score-great { background:#e1f5ee; border:1px solid #9FE1CB; color:#0F6E56; padding:14px 20px; border-radius:10px; text-align:center; }
.score-ok    { background:#faeeda; border:1px solid #FAC775; color:#BA7517; padding:14px 20px; border-radius:10px; text-align:center; }
.score-poor  { background:#faece7; border:1px solid #F5C4B3; color:#993C1D; padding:14px 20px; border-radius:10px; text-align:center; }

.flashcard   { background:white; border:1px solid rgba(0,0,0,0.1); border-radius:10px; padding:14px 18px; margin:6px 0; }
.card-front  { font-weight:500; font-size:14px; margin-bottom:6px; }
.card-back   { color:#555; font-size:13px; line-height:1.6; }
.cat-tag     { font-size:11px; padding:2px 8px; border-radius:20px; margin-top:6px; display:inline-block; }
.cat-definition  { background:#e1f5ee; color:#0F6E56; }
.cat-formula     { background:#f5f4ff; color:#534AB7; }
.cat-intuition   { background:#faeeda; color:#BA7517; }
.cat-distinction { background:#faece7; color:#993C1D; }
.correct   { color:#0F6E56; font-weight:500; }
.incorrect { color:#993C1D; font-weight:500; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "mode"           not in st.session_state: st.session_state.mode = "explain"
if "history"        not in st.session_state: st.session_state.history = []
if "quiz_state"     not in st.session_state: st.session_state.quiz_state = None
if "card_revealed"  not in st.session_state: st.session_state.card_revealed = set()
if "socratic_stuck_count" not in st.session_state: st.session_state.socratic_stuck_count = 0
pipeline = load_pipeline()

MODE_ICONS = {"explain":"💡","quiz":"✏️","summarise":"📋","flashcards":"🗂️","socratic":"🤔"}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 D2L Teaching Assistant")
    st.caption("*Dive into Deep Learning* — all chapters")
    st.divider()

    st.markdown("**Mode**")
    for key, mode in MODES.items():
        active = st.session_state.mode == key
        if st.button(
            f"{MODE_ICONS[key]} **{mode.name}**",
            key=f"btn_{key}",
            use_container_width=True,
            type="primary" if active else "secondary",
        ):
            st.session_state.mode       = key
            st.session_state.history    = []
            st.session_state.socratic_stuck_count = 0
            st.session_state.quiz_state = None
            st.session_state.card_revealed = set()
            st.rerun()

    st.divider()
    top_k = st.slider("Chunks to retrieve", 3, 12, 6)
    chapter_options = ["All chapters"] + [f"{num}. {title}" for num, title in CHAPTERS]
    selected_chapter = st.sidebar.selectbox("Restrict to chapter", chapter_options)
    
    if selected_chapter == "All chapters":
        metadata_filter = None
    else:
        chapter_num = selected_chapter.split(".", 1)[0].strip()
        metadata_filter = {"chapter": chapter_num}

    st.divider()

    st.markdown("**Try asking:**")
    for q in MODES[st.session_state.mode].example_queries[:4]:
        if st.button(q, use_container_width=True, key=f"eg_{q[:25]}"):
            st.session_state["prefill"] = q
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.history       = []
        st.session_state.quiz_state    = None
        st.session_state.card_revealed = set()
        st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
mode     = MODES[st.session_state.mode]
mode_key = st.session_state.mode

st.markdown(f"## {MODE_ICONS[mode_key]} {mode.name}")
st.caption(mode.format_hint)
st.divider()


# ── Render helpers ────────────────────────────────────────────────────────────

def render_quiz(turn: dict):
    parsed  = turn.get("parsed", {})
    quiz_id = turn.get("quiz_id", "q0")
    qs      = parsed.get("questions", [])

    st.markdown(f"**{parsed.get('topic','Quiz')}** — {len(qs)} questions")

    state = st.session_state.quiz_state
    if state and state.get("graded"):
        result = state["result"]
        pct    = result["score"] / result["total"]
        cls    = "score-great" if pct >= 0.8 else ("score-ok" if pct >= 0.5 else "score-poor")
        emoji  = "🎉" if pct >= 0.8 else ("👍" if pct >= 0.5 else "📚")
        st.markdown(
            f'<div class="{cls}">{emoji} <strong>{result["score"]}/{result["total"]}</strong> ({int(pct*100)}%)</div>',
            unsafe_allow_html=True,
        )
        for fb in result["feedback"]:
            icon = "✅" if fb["is_correct"] else "❌"
            with st.expander(f'{icon} Q{fb["id"]}: {fb["question"][:60]}...'):
                if not fb["is_correct"]:
                    st.markdown(f'Your answer: <span class="incorrect">{fb["user_answer"]}</span>', unsafe_allow_html=True)
                    st.markdown(f'Correct: <span class="correct">{fb["correct_answer"]}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="correct">✓ {fb["correct_answer"]}</span>', unsafe_allow_html=True)
                st.info(fb["explanation"])
    else:
        answers = {}
        for q in qs:
            diff = q.get("difficulty", "medium")
            st.markdown(
                f'<div class="quiz-q"><span class="diff-{diff}">{diff.upper()}</span> · '
                f'{q.get("type","").replace("_"," ").title()}<br>'
                f'<strong>Q{q["id"]}. {q["question"]}</strong></div>',
                unsafe_allow_html=True,
            )
            if q["type"] == "mcq" and q.get("options"):
                opts = q["options"]
                ans  = st.radio(
                    f"Q{q['id']}",
                    [o[0] for o in opts],
                    format_func=lambda x, o=opts: next(i for i in o if i.startswith(x)),
                    key=f"mcq_{quiz_id}_{q['id']}",
                    label_visibility="collapsed",
                )
                answers[str(q["id"])] = ans
            elif q["type"] == "true_false":
                ans = st.radio(f"Q{q['id']}", ["True","False"],
                               key=f"tf_{quiz_id}_{q['id']}",
                               label_visibility="collapsed", horizontal=True)
                answers[str(q["id"])] = ans
            else:
                ans = st.text_area(f"Q{q['id']}", key=f"sa_{quiz_id}_{q['id']}",
                                   height=80, label_visibility="collapsed",
                                   placeholder="Type your answer...")
                answers[str(q["id"])] = ans

        if st.button("✅ Check answers", type="primary", key=f"grade_{quiz_id}"):
            result = pipeline.check_answers(parsed, answers)
            st.session_state.quiz_state = {"graded": True, "result": result}
            st.rerun()


def render_flashcards(parsed: dict):
    cards = parsed.get("cards", [])
    st.markdown(f"**{parsed.get('topic','Flashcards')}** — {len(cards)} cards")
    st.caption("Click 👁 to reveal the answer.")
    for card in cards:
        cid      = str(card["id"])
        revealed = cid in st.session_state.card_revealed
        cat      = card.get("category", "definition")
        col1, col2 = st.columns([10, 1])
        with col1:
            st.markdown(
                f'<div class="flashcard">'
                f'<div class="card-front">🃏 {card["front"]}</div>'
                + (f'<div class="card-back">→ {card["back"]}</div>' if revealed else "")
                + f'<span class="cat-tag cat-{cat}">{cat}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col2:
            if st.button("👁" if not revealed else "🙈", key=f"flip_{cid}"):
                if revealed: st.session_state.card_revealed.discard(cid)
                else:        st.session_state.card_revealed.add(cid)
                st.rerun()


# ── Chat history ──────────────────────────────────────────────────────────────
for turn in st.session_state.history:
    if turn["role"] == "user":
        with st.chat_message("user"):
            st.markdown(turn["content"])
    else:
        with st.chat_message("assistant", avatar="🎓"):
            m   = turn.get("mode", "explain")
            parsed = turn.get("parsed")
            if m == "quiz" and parsed:
                render_quiz(turn)
            elif m == "flashcards" and parsed:
                render_flashcards(parsed)
            else:
                st.markdown(turn["content"])

            if turn.get("sources"):
                with st.expander(f"📎 {len(turn['sources'])} sources"):
                    for s in turn["sources"]:
                        pg = f", p.{s['page']}" if s.get("page") else ""
                        st.markdown(f"`{s['source']}{pg}` · score `{s['score']}`")
                        st.caption(s["text"] + "...")
                        st.divider()

# ── Input ─────────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", None)
prompt  = st.chat_input(mode.example_queries[0]) or prefill

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Thinking..."):
            hist = st.session_state.history[:-1] if mode_key == "socratic" else []
            result = pipeline.run(
                mode_key,
                prompt,
                hist,
                top_k=top_k,
                stuck_count=st.session_state.socratic_stuck_count,
                metadata_filter=metadata_filter,   # ← new
            )
            


        raw    = result["raw"]
        parsed = result["parsed"]
        qid    = str(uuid.uuid4())[:8]
        if result.get("socratic_meta"):
                st.session_state.socratic_stuck_count = result["socratic_meta"]["stuck_count"]    
        else:
            st.markdown(raw)

        meta = result.get("socratic_meta")
        if meta:
            badge_map = {
                "continue":   ("🔵", "Keep going"),
                "stuck":      ("🟡", "Building up to a hint"),
                "understood": ("🟢", "Got it!"),
                "concluded":  ("✅", "Dialogue complete"),
            }
            icon, label = badge_map.get(meta["status"], ("⚪", meta["status"]))
            st.caption(f"{icon} {label} · turn {meta['turn_count']}/6")


        # ── 5. (Optional) disable chat input once concluded ──
        # After appending the turn to history, check if the dialogue ended:

        if mode_key == "socratic" and turn.get("socratic_meta", {}).get("is_concluded"):
            st.success("🎓 Dialogue complete! Ask about a new topic or switch modes to continue.")

        turn = {
            "role": "assistant", "content": raw,
            "parsed": parsed, "mode": mode_key,
            "sources": result["sources"], "quiz_id": qid,
            "socratic_meta": result.get("socratic_meta"),   
        }

        if mode_key == "quiz" and parsed:
            st.session_state.quiz_state = {"graded": False}
            render_quiz(turn)
        elif mode_key == "flashcards" and parsed:
            st.session_state.card_revealed = set()
            render_flashcards(parsed)
        else:
            st.markdown(raw)

        with st.expander(f"📎 {len(result['sources'])} sources"):
            for s in result["sources"]:
                pg = f", p.{s['page']}" if s.get("page") else ""
                st.markdown(f"`{s['source']}{pg}` · score `{s['score']}`")
                st.caption(s["text"] + "...")

                st.divider()
            if metadata_filter:
                st.caption(f"🔍 Search scoped to Chapter {metadata_filter['chapter']}")

    st.session_state.history.append(turn)
