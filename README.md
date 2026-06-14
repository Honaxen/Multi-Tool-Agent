# Multi-Tool Agent

A conversational AI agent that decides which tool to call — on its own.
Ask anything. The LLM figures out whether to search the web, run math, or execute Python.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Ollama](https://img.shields.io/badge/Ollama-local-black)
![FastAPI](https://img.shields.io/badge/FastAPI-REST-green)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange)

---

## What makes this different from document-agent

In `document-agent`, the programmer hard-codes the pipeline: query → retrieve → generate.
Here, **the LLM decides** which tool (if any) to call and when to stop.
This is the core idea behind ReAct-style agents.

```
User: "What is the square root of the year Python was created?"

Agent:
  Step 1 → web_search("year Python was created")   → "1991"
  Step 2 → calculate("sqrt(1991)")                 → "44.62..."
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
│   ├── __init__.py
│   ├── tools.py          — tool definitions + registry
│   ├── tool_executor.py  — parse LLM response → run tool
│   ├── agent.py          — ReAct loop (the core logic)
│   └── main.py           — CLI interface
├── api/
│   ├── __init__.py
│   └── main.py           — FastAPI REST API
├── tests/
│   ├── __init__.py
│   └── test_tools.py     — unit tests for all tools
├── app.py                — Gradio web UI
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

---

## Getting Started

```bash
git clone https://github.com/Honaxen/multi-tool-agent.git
cd multi-tool-agent
pip install -r requirements.txt
```

Make sure Ollama is running with a model pulled:

```bash
ollama serve
ollama pull llama3.2
```

Copy the env file:

```bash
cp .env.example .env
```

---

## Running the project

### CLI — interactive

```bash
python3 -m agent.main
```

### CLI — single query

```bash
python3 -m agent.main --query "What is 2 to the power of 32?"
```

### Web UI (Gradio)

```bash
python3 app.py
```

Open: http://localhost:7860

### REST API (FastAPI)

```bash
uvicorn api.main:app --reload
```

Open: http://localhost:8000/docs

### Docker

```bash
docker build -t multi-tool-agent .
docker run -p 8000:8000 multi-tool-agent
```

### Tests

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
    {"step": 1, "type": "tool_call",   "tool_name": "calculate"},
    {"step": 2, "type": "tool_result", "tool_name": "calculate"},
    {"step": 3, "type": "answer",      "content": "The square root of 1764 is 42."}
  ],
  "duration_seconds": 2.4
}
```

Available endpoints:

| Method | Endpoint | Description |
|---|---|---|
| POST | `/query` | Run the agent on a query |
| GET | `/tools` | List available tool schemas |
| GET | `/health` | Health check |

---

## Stack

| Layer | Technology |
|---|---|
| LLM | Ollama (llama3.2 — local) |
| Agent loop | Custom ReAct implementation |
| Tools | DuckDuckGo API · AST math eval · subprocess |
| API | FastAPI + Pydantic |
| UI | Gradio |
| Tests | pytest |
| Container | Docker |

---

## What I Learned

A model predicts. An agent **reasons**.

The key insight: when the LLM's response is valid JSON with a `"tool"` key,
we treat it as a command. When it's plain text, we treat it as the final answer.
That one check is the entire difference between a chatbot and an agent.

Tool schemas are instructions, not bindings. The LLM reads the schema descriptions
in plain English and decides what to call. Well-written descriptions matter more
than the tool implementation itself.

The ReAct loop is simpler than it sounds:
think → act → observe → think again. That's it.

---

## Related projects

- [nlp-intelligence-lab](https://github.com/Honaxen/nlp-intelligence-lab) — NLP foundations
- [semantic-similarity-engine](https://github.com/Honaxen/semantic-similarity-engine) — embeddings + similarity
- [rag-system-from-scratch](https://github.com/Honaxen/rag-system-from-scratch) — RAG pipeline
- [document-agent](https://github.com/Honaxen/document-agent) — previous agent project

---

## Author

[Honaxen](https://github.com/Honaxen)