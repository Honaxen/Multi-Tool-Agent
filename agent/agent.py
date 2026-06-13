"""
agent.py — The main ReAct loop.

Flow per query:
  1. Send query + tool schemas to Ollama
  2. LLM either answers directly OR returns a tool call JSON
  3. If tool call → execute → send result back to LLM
  4. Repeat up to MAX_STEPS times
  5. Return final answer

This is the key difference from document-agent:
  The LLM decides which tool (if any) to call — not the programmer.
"""

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from agent.tools import TOOLS
from agent.tool_executor import run_tool_call, ToolCallError

# ── Config ───────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.2"   # change to any model you have pulled
MAX_STEPS = 6                # max tool calls per query


# ── Build the system prompt ──────────────────

def _build_system_prompt() -> str:
    schemas = [t["schema"] for t in TOOLS.values()]
    schemas_json = json.dumps(schemas, indent=2)

    return f"""You are a helpful AI assistant with access to the following tools:

{schemas_json}

## How to use tools

When you need to use a tool, respond with ONLY a JSON object in this exact format:
{{
  "tool": "<tool_name>",
  "args": {{
    "<arg_name>": "<arg_value>"
  }}
}}

Do NOT add any text before or after the JSON when calling a tool.

After receiving a tool result, you may call another tool OR provide your final answer as plain text.

When you have enough information, respond with a plain text answer — do NOT wrap it in JSON.

## Rules
- Use web_search for current facts, news, or anything you're unsure about.
- Use calculate for any math — never do arithmetic in your head.
- Use run_python for data processing, generating lists, string manipulation, or anything needing real computation.
- Be concise. Don't repeat the tool result verbatim — summarize and explain.
"""


# ── Ollama client ────────────────────────────

def _call_ollama(messages: list[dict], model: str) -> str:
    """Send messages to Ollama and return the assistant's response text."""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
            return data["message"]["content"]
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Cannot reach Ollama at {OLLAMA_URL}. "
            f"Is it running? Run: ollama serve\nError: {e}"
        )


# ── Step result dataclass ────────────────────

@dataclass
class Step:
    type: str          # "tool_call" | "tool_result" | "answer"
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[str] = None


# ── Main agent ───────────────────────────────

@dataclass
class AgentResponse:
    answer: str
    steps: list[Step] = field(default_factory=list)
    model: str = DEFAULT_MODEL


def _is_tool_call(text: str) -> bool:
    """Heuristic: does this look like a tool call JSON?"""
    stripped = text.strip().lstrip("```json").lstrip("```").strip()
    if not stripped.startswith("{"):
        return False
    try:
        data = json.loads(stripped if not stripped.endswith("```") else stripped[:-3])
        return "tool" in data and "args" in data
    except json.JSONDecodeError:
        return False


def run_agent(query: str, model: str = DEFAULT_MODEL) -> AgentResponse:
    """
    Run the full ReAct loop for a single query.
    Returns the final answer and a trace of all steps taken.
    """
    system_prompt = _build_system_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": query},
    ]
    steps: list[Step] = []

    for step_num in range(MAX_STEPS):
        response = _call_ollama(messages, model)

        if _is_tool_call(response):
            # ── Tool call branch ──
            steps.append(Step(
                type="tool_call",
                content=response,
                tool_name=None,
                tool_args=None,
            ))

            try:
                tool_name, args_str, result = run_tool_call(response)
                steps[-1].tool_name = tool_name
                steps[-1].tool_args = args_str
                result_str = str(result)

            except ToolCallError as e:
                result_str = f"Tool error: {e}"
                tool_name = "unknown"

            steps.append(Step(
                type="tool_result",
                content=result_str,
                tool_name=tool_name,
            ))

            # Feed result back to LLM
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Tool result for {tool_name}:\n{result_str}\n\nContinue."
            })

        else:
            # ── Final answer branch ──
            steps.append(Step(type="answer", content=response))
            return AgentResponse(answer=response, steps=steps, model=model)

    # Fallback if MAX_STEPS reached without a plain answer
    fallback = "I reached the maximum number of steps without a final answer. Please try rephrasing your question."
    steps.append(Step(type="answer", content=fallback))
    return AgentResponse(answer=fallback, steps=steps, model=model)
