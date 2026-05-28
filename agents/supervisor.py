"""Multi-agent supervisor orchestrator built with LangGraph."""

from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import create_react_agent

from agents.llm import get_llm
from memory.checkpointer import get_checkpointer
from tools.tools import get_tools

# ---------------------------------------------------------------------------
# Worker definitions
# ---------------------------------------------------------------------------

WORKER_CONFIGS = {
    "researcher": "You are a research assistant. Search for information and summarize findings.",
    "writer": "You are a writing assistant. Draft clear, concise content based on research provided.",
}


def _build_worker(name: str, system_prompt: str):
    """Build a named ReAct worker agent."""
    llm = get_llm()
    tools = get_tools()
    agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)

    def node(state: MessagesState):
        result = agent.invoke(state)
        last = result["messages"][-1]
        return {"messages": [HumanMessage(content=last.content, name=name)]}

    return node


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

WorkerName = Literal["researcher", "writer", "FINISH"]

SUPERVISOR_SYSTEM = (
    "You are a supervisor managing the following workers: {workers}.\n"
    "Given the conversation, decide which worker should act next, or reply FINISH "
    "when the task is complete.\n"
    "Reply with only the worker name or FINISH."
)


def _supervisor_node(state: MessagesState):
    """Decide which worker should run next, or finish."""
    llm = get_llm()
    workers = list(WORKER_CONFIGS.keys())
    prompt = SUPERVISOR_SYSTEM.format(workers=", ".join(workers))
    messages = [{"role": "system", "content": prompt}] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def _route(state: MessagesState) -> str:
    last_content = state["messages"][-1].content.strip()
    if last_content == "FINISH":
        return END
    if last_content in WORKER_CONFIGS:
        return last_content
    return END


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def create_supervisor_agent():
    """Create and return a compiled multi-agent supervisor graph."""
    builder = StateGraph(MessagesState)

    # Add supervisor node
    builder.add_node("supervisor", _supervisor_node)

    # Add worker nodes
    for name, prompt in WORKER_CONFIGS.items():
        builder.add_node(name, _build_worker(name, prompt))
        builder.add_edge(name, "supervisor")

    # Routing
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", _route)

    checkpointer = get_checkpointer()
    return builder.compile(checkpointer=checkpointer)
