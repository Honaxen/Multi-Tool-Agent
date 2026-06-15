"""
tools.py — The 3 tools the agent can call.

Each tool is a plain Python function.
The schema dict tells the LLM what each tool does and what args it takes.
"""

import ast
import math
import subprocess
import urllib.parse
import urllib.request
import json
import tempfile
import os
from typing import Any


# ─────────────────────────────────────────────
# Tool 1: Web Search  (DuckDuckGo Instant API)
# ─────────────────────────────────────────────

def web_search(query: str) -> str:
    """Search for information using Wikipedia and DuckDuckGo APIs."""
    import ssl
    import re as _re

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    # ── Try 1: Wikipedia search API ──────────────────
    try:
        encoded = urllib.parse.quote_plus(query)
        wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded}&format=json&srlimit=3"
        req = urllib.request.Request(wiki_url, headers={"User-Agent": "multi-tool-agent/1.0 (educational project)"})
        with urllib.request.urlopen(req, timeout=8, context=ssl_ctx) as resp:
            data = json.loads(resp.read().decode())

        hits = data.get("query", {}).get("search", [])
        if hits:
            # Get the top result's extract
            title = hits[0]["title"]
            extract_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=true&explaintext=true&titles={urllib.parse.quote_plus(title)}&format=json"
            req2 = urllib.request.Request(extract_url, headers={"User-Agent": "multi-tool-agent/1.0"})
            with urllib.request.urlopen(req2, timeout=8, context=ssl_ctx) as resp2:
                edata = json.loads(resp2.read().decode())
            pages = edata.get("query", {}).get("pages", {})
            for page in pages.values():
                extract = page.get("extract", "")
                if extract:
                    # Return first 500 chars
                    return extract[:500].strip()
    except Exception:
        pass

    # ── Try 2: DuckDuckGo Instant Answer API ─────────
    try:
        encoded = urllib.parse.quote_plus(query)
        url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_redirect=1&no_html=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"})
        with urllib.request.urlopen(req, timeout=8, context=ssl_ctx) as resp:
            data = json.loads(resp.read().decode())

        if data.get("AbstractText"):
            return data["AbstractText"]

        topics = data.get("RelatedTopics", [])
        results = [t["Text"] for t in topics[:3] if isinstance(t, dict) and t.get("Text")]
        if results:
            return "
".join(results)

    except Exception:
        pass

    return f"Could not retrieve results for: \"{query}\". The agent will answer from its training knowledge."
# ─────────────────────────────────────────────
# Tool 2: Calculator  (safe math eval)
# ─────────────────────────────────────────────

# Whitelist of safe names for the evaluator
_SAFE_NAMES = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
_SAFE_NAMES.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})


def _safe_eval(expr: str) -> Any:
    """Parse and evaluate a math expression using AST — no exec/eval with globals."""
    tree = ast.parse(expr, mode="eval")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in _SAFE_NAMES:
                return _SAFE_NAMES[node.id]
            raise ValueError(f"Name not allowed: {node.id}")
        elif isinstance(node, ast.BinOp):
            ops = {
                ast.Add: lambda a, b: a + b,
                ast.Sub: lambda a, b: a - b,
                ast.Mult: lambda a, b: a * b,
                ast.Div: lambda a, b: a / b,
                ast.Pow: lambda a, b: a ** b,
                ast.Mod: lambda a, b: a % b,
                ast.FloorDiv: lambda a, b: a // b,
            }
            op_fn = ops.get(type(node.op))
            if not op_fn:
                raise ValueError(f"Operator not allowed: {node.op}")
            return op_fn(_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -_eval(node.operand)
            elif isinstance(node.op, ast.UAdd):
                return +_eval(node.operand)
        elif isinstance(node, ast.Call):
            fn = _eval(node.func)
            args = [_eval(a) for a in node.args]
            return fn(*args)
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    return _eval(tree)


def calculate(expression: str) -> str:
    """Evaluate a math expression safely. Supports +,-,*,/,**,%, math functions."""
    try:
        result = _safe_eval(expression.strip())
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        return f"Calculation error: {e}"


# ─────────────────────────────────────────────
# Tool 3: Python Runner  (subprocess sandbox)
# ─────────────────────────────────────────────

_BLOCKED_IMPORTS = [
    "os.system", "subprocess", "shutil.rmtree",
    "open('/etc", "open('/root", "__import__",
]


def run_python(code: str) -> str:
    """Run a short Python snippet and return its stdout. 10s timeout, no network."""
    # Basic safety check
    for blocked in _BLOCKED_IMPORTS:
        if blocked in code:
            return f"Blocked: '{blocked}' is not allowed in sandboxed execution."

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "PYTHONPATH": ""},  # clean env
        )
        output = result.stdout.strip()
        error = result.stderr.strip()

        if error and not output:
            return f"Error:\n{error}"
        if error:
            return f"Output:\n{output}\n\nStderr:\n{error}"
        return output or "(no output)"

    except subprocess.TimeoutExpired:
        return "Error: code timed out after 10 seconds."
    except Exception as e:
        return f"Execution error: {e}"
    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────
# Tool Registry — what the LLM sees
# ─────────────────────────────────────────────

TOOLS = {
    "web_search": {
        "fn": web_search,
        "schema": {
            "name": "web_search",
            "description": "Search the web for current information. Use when asked about recent events, facts, or anything you are uncertain about.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, written as a natural search phrase."
                    }
                },
                "required": ["query"]
            }
        }
    },
    "calculate": {
        "fn": calculate,
        "schema": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression. Use for any arithmetic, algebra, or math functions (sqrt, sin, log, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A valid Python math expression, e.g. '2 ** 10', 'sqrt(144)', '(3 + 4) * 2'."
                    }
                },
                "required": ["expression"]
            }
        }
    },
    "run_python": {
        "fn": run_python,
        "schema": {
            "name": "run_python",
            "description": "Execute a short Python code snippet and return its stdout. Use for data processing, list manipulation, string operations, or anything needing real computation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Valid Python 3 code. Must print output to stdout to see results."
                    }
                },
                "required": ["code"]
            }
        }
    }
}