"""
socratic_graph.py — LangGraph state machine for Socratic tutoring.

Why a graph instead of a plain prompt-with-history chain?

The other 4 modes (Explain, Quiz, Summarise, Flashcards) are single-turn:
one question in, one answer out. A linear LCEL chain (prompt | llm | parser)
is the right tool for that.

Socratic is different — it's a *process* with real decision points:
  - Did the student actually understand, or are they still confused?
  - Have they been stuck for multiple turns and need a hint?
  - Should the dialogue end now, or keep going?

Those are branching decisions, not just "generate the next message."
A plain prompt can be *told* to behave this way, but it has no memory
of turn count and no hard guarantee it will ever conclude. A graph makes
the control flow explicit and enforceable in code, not just in English
inside a system prompt.

Graph shape (one run of the graph = one user turn):

              START
                │
                ▼
         ┌─────────────┐
         │  retrieve   │   fetch passages relevant to the topic
         └─────────────┘
                │
                ▼
         ┌─────────────┐
         │  evaluate   │   classify the student's last reply
         └─────────────┘   (skipped on the very first turn)
                │
      ┌─────────┼─────────────┐
      ▼         ▼              ▼
  [understood] [stuck≥2]   [continue]
      │         │              │
      ▼         ▼              ▼
 ┌─────────┐ ┌──────┐    ┌──────────┐
 │conclude │ │ hint │    │ respond  │
 └─────────┘ └──────┘    └──────────┘
      │         │              │
      ▼         ▼              ▼
     END       END            END

pip install langgraph langchain-openai langchain-core
"""

from __future__ import annotations
from typing import TypedDict, Literal, Annotated
import operator

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


# ------------------------------------------------------------------------ #
#  STATE                                                                     #
# ------------------------------------------------------------------------ #

class SocraticState(TypedDict):
    topic: str                              # the concept being explored
    context: str                            # retrieved book passages
    messages: Annotated[list[BaseMessage], operator.add]   # full dialogue so far
    turn_count: int                         # how many tutor questions asked
    stuck_count: int                        # consecutive "confused" replies
    status: Literal["continue", "understood", "stuck", "concluded"]
    final_response: str                     # what gets shown to the user this turn


MAX_TURNS = 6          # hard cap so a dialogue can't run forever
STUCK_THRESHOLD = 2    # give a hint after this many consecutive confused replies


# ------------------------------------------------------------------------ #
#  PROMPTS                                                                    #
# ------------------------------------------------------------------------ #

SOCRATIC_SYSTEM = """You are a Socratic tutor for "Dive into Deep Learning".

Rules — follow these strictly:
- NEVER state the answer directly, even when the student is close
- Ask ONE question at a time — never a list of questions
- Start broad, narrow based on their response
- When they get something right, acknowledge briefly then push deeper
- When they're wrong, don't say "wrong" — ask a question that reveals the gap
- Keep responses to 2-4 sentences — this is a dialogue, not a lecture

Use the source passages below as your knowledge base to check the student's
reasoning and to steer questions toward the real content.

<source_passages>
{context}
</source_passages>"""

EVALUATE_SYSTEM = """You are grading a Socratic dialogue turn.

Topic: {topic}

Look at the student's most recent reply in the conversation below and
classify it into exactly one category:

- "understood" — the student has clearly grasped the core idea and could
                  explain it correctly in their own words
- "stuck"       — the student is confused, gave a wrong answer, or said
                  they don't know
- "continue"    — the student is on the right track but hasn't fully
                  arrived at the core idea yet

Respond with ONLY one word: understood, stuck, or continue."""

HINT_SYSTEM = """You are a Socratic tutor for "Dive into Deep Learning".

The student has been stuck for a couple of turns. Give ONE small, concrete
hint — not the full answer — that nudges them toward the idea. Then ask
a simpler, more specific question than before.

Keep it to 2-3 sentences.

<source_passages>
{context}
</source_passages>"""

CONCLUDE_SYSTEM = """You are a Socratic tutor for "Dive into Deep Learning".

The student has demonstrated understanding of the topic. Write a brief,
warm closing message (2-3 sentences):
  1. Confirm what they got right, restating it in clean terms
  2. One sentence connecting it to why it matters or what it enables

Do not ask another question — this is the end of the dialogue."""


# ------------------------------------------------------------------------ #
#  GRAPH BUILDER                                                              #
# ------------------------------------------------------------------------ #

def build_socratic_graph(llm):
    """
    Build and compile the LangGraph state machine.
    `llm` is a LangChain ChatOpenAI instance, injected from the pipeline
    so the graph reuses the same client rather than creating its own.
    """

    # -- Node: retrieve is handled OUTSIDE the graph (pipeline already
    #    has a HybridRetriever) — context is passed in via initial state.
    #    We keep an evaluate/respond/hint/conclude graph here.

    def evaluate_node(state: SocraticState) -> dict:
        """Classify the student's last reply. Skipped on turn 0."""
        if state["turn_count"] == 0:
            return {"status": "continue"}

        prompt = ChatPromptTemplate.from_messages([
            ("system", EVALUATE_SYSTEM),
            MessagesPlaceholder(variable_name="messages"),
        ])
        chain = prompt | llm | StrOutputParser()

        verdict = chain.invoke({
            "topic": state["topic"],
            "messages": state["messages"],
        }).strip().lower()

        if "understood" in verdict:
            return {"status": "understood"}
        elif "stuck" in verdict:
            return {"status": "stuck", "stuck_count": state["stuck_count"] + 1}
        else:
            return {"status": "continue", "stuck_count": 0}

    def route_after_evaluate(state: SocraticState) -> str:
        """Conditional edge — decides which node runs next."""
        if state["turn_count"] >= MAX_TURNS:
            return "conclude"
        if state["status"] == "understood":
            return "conclude"
        if state["status"] == "stuck" and state["stuck_count"] >= STUCK_THRESHOLD:
            return "hint"
        return "respond"

    def respond_node(state: SocraticState) -> dict:
        """Ask the next Socratic question."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", SOCRATIC_SYSTEM),
            MessagesPlaceholder(variable_name="messages"),
        ])
        chain = prompt | llm | StrOutputParser()

        response = chain.invoke({
            "context": state["context"],
            "messages": state["messages"],
        })

        return {
            "messages": [AIMessage(content=response)],
            "turn_count": state["turn_count"] + 1,
            "final_response": response,
        }

    def hint_node(state: SocraticState) -> dict:
        """Give a small hint after the student is stuck for a while."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", HINT_SYSTEM),
            MessagesPlaceholder(variable_name="messages"),
        ])
        chain = prompt | llm | StrOutputParser()

        response = chain.invoke({
            "context": state["context"],
            "messages": state["messages"],
        })

        return {
            "messages": [AIMessage(content=response)],
            "turn_count": state["turn_count"] + 1,
            "stuck_count": 0,   # reset after giving a hint
            "final_response": response,
        }

    def conclude_node(state: SocraticState) -> dict:
        """Wrap up the dialogue once the student understands (or turns run out)."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", CONCLUDE_SYSTEM),
            MessagesPlaceholder(variable_name="messages"),
        ])
        chain = prompt | llm | StrOutputParser()

        response = chain.invoke({"messages": state["messages"]})

        return {
            "messages": [AIMessage(content=response)],
            "status": "concluded",
            "final_response": response,
        }

    # -- Assemble the graph --------------------------------------------- #
    graph = StateGraph(SocraticState)

    graph.add_node("evaluate", evaluate_node)
    graph.add_node("respond",  respond_node)
    graph.add_node("hint",     hint_node)
    graph.add_node("conclude", conclude_node)

    graph.set_entry_point("evaluate")

    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"respond": "respond", "hint": "hint", "conclude": "conclude"},
    )

    graph.add_edge("respond",  END)
    graph.add_edge("hint",     END)
    graph.add_edge("conclude", END)

    return graph.compile()


# ------------------------------------------------------------------------ #
#  PUBLIC WRAPPER                                                            #
# ------------------------------------------------------------------------ #

class SocraticSession:
    """
    Thin wrapper the pipeline calls into.
    Owns the compiled graph and converts to/from the pipeline's
    plain-dict history format.
    """

    def __init__(self, llm):
        self._graph = build_socratic_graph(llm)

    def run(
        self,
        topic: str,
        context: str,
        history: list[dict],   # [{"role": "user"/"assistant", "content": str}]
        user_message: str,
        stuck_count: int = 0,  # carried in from the caller's session state
    ) -> dict:
        """
        Run one turn of the Socratic graph.

        stuck_count must be passed in and persisted by the caller (e.g. in
        st.session_state) because each call to run() invokes the graph
        fresh — LangGraph's in-memory state doesn't survive between
        separate Streamlit reruns. Without carrying this forward, the
        hint node would never trigger since every turn would look like
        the student's first "stuck" reply.

        Returns:
          {
            "response": str,          # text to show the student
            "status": str,            # continue | understood | stuck | concluded
            "turn_count": int,
            "stuck_count": int,       # feed this back in on the next call
            "is_concluded": bool,
          }
        """
        # Convert prior history into LangChain messages
        lc_messages: list[BaseMessage] = []
        for turn in history:
            if turn["role"] == "user":
                lc_messages.append(HumanMessage(content=turn["content"]))
            else:
                lc_messages.append(AIMessage(content=turn["content"]))

        # Append the new user turn
        lc_messages.append(HumanMessage(content=user_message))

        # turn_count = number of AI messages already sent
        turn_count = sum(1 for m in lc_messages if isinstance(m, AIMessage))

        initial_state: SocraticState = {
            "topic":          topic,
            "context":        context,
            "messages":       lc_messages,
            "turn_count":     turn_count,
            "stuck_count":    stuck_count,
            "status":         "continue",
            "final_response": "",
        }

        result = self._graph.invoke(initial_state)

        return {
            "response":     result["final_response"],
            "status":       result["status"],
            "turn_count":   result["turn_count"],
            "stuck_count":  result["stuck_count"],
            "is_concluded": result["status"] == "concluded",
        }
