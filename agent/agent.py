"""
agent.py — The main ReAct loop.
"""

import json
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from agent.tools import TOOLS
from agent.tool_executor import run_tool_call, ToolCallError

# ── Config ───────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "gemma3:12b"
MAX_STEPS = 8


# ── Build the system prompt ──────────────────

def _build_system_prompt() -> str:
    schemas = [t["schema"] for t in TOOLS.values()]
    schemas_json = json.dumps(schemas, indent=2)

    return f"""You are a helpful AI assistant with access to tools.

AVAILABLE TOOLS:
{schemas_json}

STRICT RULES — READ CAREFULLY:
1. ONE action per response. Either call ONE tool OR give a final answer. Never both.
2. To call a tool, output ONLY this JSON with no text before or after:
{{"tool": "<tool_name>", "args": {{"<arg_name>": "<arg_value>"}}}}

3. NEVER do math in your head — always use the calculate tool.
4. ALWAYS use web_search for facts about people, places, history, or current events.
5. After receiving a tool result, you may call another tool OR give your final plain text answer.
6. Your final answer must be plain text only — NO JSON, NO tool calls.

EXAMPLES:
User: What is 2^10?
You: {{"tool": "calculate", "args": {{"expression": "2 ** 10"}}}}
[tool result: 2 ** 10 = 1024]
You: 2 to the power of 10 is 1024.

User: What year was Python created? Then calculate sqrt of that year.
You: {{"tool": "web_search", "args": {{"query": "year Python programming language created"}}}}
[tool result: Python was first released in 1991.]
You: {{"tool": "calculate", "args": {{"expression": "sqrt(1991)"}}}}
[tool result: sqrt(1991) = 44.62]
You: Python was created in 1991. The square root of 1991 is approximately 44.62.
"""


# ── Ollama client ────────────────────────────

def _call_ollama(messages: list[dict], model: str) -> str:
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
    type: str
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[str] = None


@dataclass
class AgentResponse:
    answer: str
    steps: list[Step] = field(default_factory=list)
    model: str = DEFAULT_MODEL


# ── Tool call detection ──────────────────────

def _extract_json(text: str) -> Optional[str]:
    """
    Extract a JSON tool call from LLM response.
    Handles: raw JSON, fenced blocks, JSON embedded in text.
    """
    stripped = text.strip()

    # Try 1: pure JSON
    if stripped.startswith("{"):
        candidate = stripped.split("```")[0].strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # Try 2: fenced block ```json ... ``` or ```tool ... ```
    match = re.search(r"```(?:json|tool)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try 3: any {"tool": ..., "args": ...} anywhere in the text
    match = re.search(r'\{\s*"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{[^{}]*\}\s*\}', stripped, re.DOTALL)
    if match:
        return match.group(0).strip()

    return None


def _is_tool_call(text: str) -> bool:
    candidate = _extract_json(text)
    if not candidate:
        return False
    try:
        data = json.loads(candidate)
        return "tool" in data and "args" in data
    except json.JSONDecodeError:
        return False


# ── Main agent loop ──────────────────────────

def run_agent(query: str, model: str = DEFAULT_MODEL) -> AgentResponse:
    system_prompt = _build_system_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": query},
    ]
    steps: list[Step] = []

    for step_num in range(MAX_STEPS):
        response = _call_ollama(messages, model)

        if _is_tool_call(response):
            steps.append(Step(type="tool_call", content=response))

            clean_json = _extract_json(response)
            try:
                tool_name, args_str, result = run_tool_call(clean_json)
                steps[-1].tool_name = tool_name
                steps[-1].tool_args = args_str
                result_str = str(result)
            except ToolCallError as e:
                result_str = f"Tool error: {e}"
                tool_name = "unknown"

            steps.append(Step(type="tool_result", content=result_str, tool_name=tool_name))

            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": (
                    f"Tool result for {tool_name}:\n{result_str}\n\n"
                    "If you need another tool, call it now. "
                    "Otherwise give your final plain text answer with NO JSON."
                )
            })

        else:
            steps.append(Step(type="answer", content=response))
            return AgentResponse(answer=response, steps=steps, model=model)

    fallback = "Reached maximum steps without a final answer. Please rephrase your question."
    steps.append(Step(type="answer", content=fallback))
    return AgentResponse(answer=fallback, steps=steps, model=model)