# Multi-Tool Agent

A conversational AI agent that decides which tool to call — on its own.
Ask anything. The LLM figures out whether to search the web, run math, or execute Python.

---

## What makes this different from document-agent

In `document-agent`, the programmer hard-codes the pipeline: query → retrieve → generate.
Here, **the LLM decides** which tool (if any) to call and when to stop.
This is the core idea behind ReAct-style agents.

```
User: "What is the square root of the year Python was created?"

Agent:
  Step 1 → web_search("year Python was created")      → "1991"
  Step 2 → calculate("sqrt(1991)")                    → "44.62..."
  Answer → "Python was created in 1991. √1991 ≈ 44.62"
```

---

## Tools

| Tool | Description | When the LLM uses it |
|---|---|---|
| `web_search` | DuckDuckGo Instant Answer API | Current facts, news, uncertain knowledge |
| `calculate` | Safe AST-based math evaluator | Any arithmetic or math function |
| `run_python` | Sandboxed subprocess execution | Data processing, list generation, string ops |

---

## Project Structure

```
multi-tool-agent/
├── agent/
│   ├── tools.py          — tool definitions + registry
│   ├── tool_executor.py  — parse LLM response → run tool
│   ├── agent.py          — ReAct loop (the core logic)
│   └── main.py           — CLI interface
├── api/
│   └── main.py           — FastAPI REST API
├── tests/
│   └── test_tools.py     — unit tests for all tools
├── app.py                — Gradio web UI
├── requirements.txt
├── Dockerfile
└── .env.example
```

---

## Getting Started

```bash
pip install -r requirements.txt
```

Make sure Ollama is running with a model pulled:

```bash
ollama serve
ollama pull llama3.2
```

### Run CLI (interactive)

```bash
python3 -m agent.main
```

### Run CLI (single query)

```bash
python3 -m agent.main --query "What is 2 to the power of 32?"
```

### Run Web UI (Gradio)

```bash
python3 app.py
```

Open: http://localhost:7860

### Run API (FastAPI)

```bash
uvicorn api.main:app --reload
```

Open: http://localhost:8000/docs

### Run Tests

```bash
pytest tests/ -v
```

---

## API Usage

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is sqrt(1764)?"}'
```

Response:
```json
{
  "query": "What is sqrt(1764)?",
  "answer": "The square root of 1764 is 42.",
  "model": "llama3.2",
  "steps": [
    {"step": 1, "type": "tool_call",   "tool_name": "calculate", ...},
    {"step": 2, "type": "tool_result", "tool_name": "calculate", ...},
    {"step": 3, "type": "answer",      "content": "The square root..."}
  ],
  "duration_seconds": 2.4
}
```

---

## Stack

Python · Ollama · FastAPI · Gradio · pytest

---

## What I Learned

A model predicts. An agent **reasons**.

The key insight: when the LLM's response is valid JSON with a `"tool"` key,
we treat it as a command. When it's plain text, we treat it as the final answer.
That one check is the entire difference between a chatbot and an agent.

Tool schemas are instructions, not bindings.
The LLM reads the schema descriptions in plain English and decides what to call.
Well-written descriptions matter more than the tool implementation itself.

The ReAct loop is simpler than it sounds:
think → act → observe → think again. That's it.

---

## Author

[Honaxen](https://github.com/Honaxen)