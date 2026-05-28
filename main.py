"""Application entry point.

Run locally:
    uvicorn main:app --reload

Or via the helper script:
    python main.py
"""

import uvicorn
from fastapi import FastAPI

from api.routes import router

app = FastAPI(
    title="AI Agent Boilerplate",
    description="A starter FastAPI + LangGraph agent service.",
    version="0.1.0",
)

app.include_router(router)


@app.get("/")
def root():
    """Root health-check."""
    return {"message": "AI Agent Boilerplate is running. Visit /docs for the API."}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
