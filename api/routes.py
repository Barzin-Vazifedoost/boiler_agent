"""FastAPI router exposing agent endpoints."""

import uuid
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/agent", tags=["agent"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None  # generated server-side when not provided
    mode: Optional[str] = "worker"  # "worker" | "supervisor"


class ChatResponse(BaseModel):
    reply: str
    thread_id: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_agent(agent, message: str, thread_id: str) -> str:
    """Invoke an agent and return the last assistant message as a string."""
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config=config)
    return result["messages"][-1].content


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Send a message to the worker or supervisor agent and get a reply.

    - **message**: The user message.
    - **thread_id**: Identifies the conversation thread (used for memory).
      A unique ID is generated if not provided.
    - **mode**: `"worker"` (default) for a single ReAct agent,
                `"supervisor"` for the multi-agent orchestrator.
    """
    thread_id = request.thread_id or str(uuid.uuid4())

    if request.mode == "supervisor":
        from agents.supervisor import create_supervisor_agent

        agent = create_supervisor_agent()
    else:
        from agents.worker import create_worker_agent

        agent = create_worker_agent()

    reply = _run_agent(agent, request.message, thread_id)
    return ChatResponse(reply=reply, thread_id=thread_id)


@router.get("/health")
def health():
    """Health-check endpoint."""
    return {"status": "ok"}
