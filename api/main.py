"""
api/main.py — FastAPI REST interface for the multi-tool agent.

Endpoints:
  POST /query      — run the agent on a query
  GET  /tools      — list available tools
  GET  /health     — health check

Run:
    uvicorn api.main:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import time

from agent.agent import run_agent, DEFAULT_MODEL, AgentResponse
from agent.tools import TOOLS

app = FastAPI(
    title="Multi-Tool Agent API",
    description="An LLM agent that decides which tool to call — web search, calculator, or Python runner.",
    version="1.0.0",
)


# ── Request / Response models ────────────────

class QueryRequest(BaseModel):
    query: str
    model: Optional[str] = DEFAULT_MODEL


class StepOut(BaseModel):
    step: int
    type: str
    tool_name: Optional[str] = None
    tool_args: Optional[str] = None
    content: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    model: str
    steps: list[StepOut]
    duration_seconds: float


# ── Endpoints ────────────────────────────────

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Run the agent on a query and return the answer with full step trace."""
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Query cannot be empty.")

    start = time.time()
    try:
        response: AgentResponse = run_agent(request.query, model=request.model)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    duration = round(time.time() - start, 2)

    steps_out = [
        StepOut(
            step=i + 1,
            type=s.type,
            tool_name=s.tool_name,
            tool_args=s.tool_args,
            content=s.content,
        )
        for i, s in enumerate(response.steps)
    ]

    return QueryResponse(
        query=request.query,
        answer=response.answer,
        model=response.model,
        steps=steps_out,
        duration_seconds=duration,
    )


@app.get("/tools")
def list_tools():
    """Return the schemas of all available tools."""
    return {
        "tools": [t["schema"] for t in TOOLS.values()]
    }


@app.get("/health")
def health():
    return {"status": "ok"}
