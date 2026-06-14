"""
tool_executor.py — Parses the LLM's tool_call response and runs it.

The LLM returns JSON like:
  {"tool": "calculate", "args": {"expression": "sqrt(144)"}}

This module validates, dispatches, and returns the result.
"""

import json
from typing import Any
from agent.tools import TOOLS


class ToolCallError(Exception):
    pass


def parse_tool_call(raw: str) -> tuple[str, dict]:
    """
    Extract tool name and args from the LLM's raw response string.
    The LLM is instructed to respond with a JSON block when calling a tool.
    """
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first fence line (```json or ```)
        inner = lines[1:] if len(lines) > 1 else lines
        # Remove last fence line if it's just ```
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        cleaned = "\n".join(inner).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ToolCallError(f"Could not parse tool call JSON: {e}\nRaw: {raw!r}")

    tool_name = data.get("tool")
    args = data.get("args", {})

    if not tool_name:
        raise ToolCallError(f"No 'tool' key in response: {data}")

    if tool_name not in TOOLS:
        raise ToolCallError(f"Unknown tool: '{tool_name}'. Available: {list(TOOLS.keys())}")

    return tool_name, args


def execute_tool(tool_name: str, args: dict) -> Any:
    """Call the tool function with the provided args."""
    tool_fn = TOOLS[tool_name]["fn"]
    try:
        return tool_fn(**args)
    except TypeError as e:
        raise ToolCallError(f"Bad args for '{tool_name}': {e}")


def run_tool_call(raw_response: str) -> tuple[str, str, Any]:
    """
    Full pipeline: parse → execute → return.
    Returns (tool_name, args_str, result).
    """
    tool_name, args = parse_tool_call(raw_response)
    result = execute_tool(tool_name, args)
    args_str = json.dumps(args)
    return tool_name, args_str, result