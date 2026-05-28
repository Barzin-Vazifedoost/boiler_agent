"""Single ReAct worker agent built with LangGraph."""

from langgraph.prebuilt import create_react_agent

from agents.llm import get_llm
from memory.checkpointer import get_checkpointer
from tools.tools import get_tools


def create_worker_agent(system_prompt: str = "You are a helpful assistant."):
    """Create and return a compiled ReAct agent graph.

    Args:
        system_prompt: The system message given to the agent.

    Returns:
        A compiled LangGraph StateGraph that can be invoked or streamed.
    """
    llm = get_llm()
    tools = get_tools()
    checkpointer = get_checkpointer()

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        checkpointer=checkpointer,
    )
    return agent
