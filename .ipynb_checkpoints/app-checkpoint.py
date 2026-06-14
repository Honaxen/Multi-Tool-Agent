"""
app.py — Gradio web UI for the multi-tool agent.

Run:
    python3 app.py
Then open: http://localhost:7860
"""

import gradio as gr
from agent.agent import run_agent, DEFAULT_MODEL
from agent.tools import TOOLS

TOOL_NAMES = list(TOOLS.keys())


def format_steps(steps) -> str:
    """Format step trace as readable markdown."""
    lines = []
    for i, step in enumerate(steps, 1):
        if step.type == "tool_call":
            lines.append(f"**Step {i} — Tool call: `{step.tool_name}`**")
            if step.tool_args:
                lines.append(f"```json\n{step.tool_args}\n```")
        elif step.type == "tool_result":
            lines.append(f"**Step {i} — Result from `{step.tool_name}`**")
            content = step.content
            if len(content) > 500:
                content = content[:500] + "\n*(truncated)*"
            lines.append(f"```\n{content}\n```")
        # Skip "answer" steps — shown in main output box
    return "\n\n".join(lines) if lines else "*No tool calls made — answered directly.*"


def run_query(query: str, model: str):
    if not query.strip():
        return "Please enter a query.", ""

    try:
        response = run_agent(query, model=model)
    except ConnectionError:
        return (
            "❌ Cannot reach Ollama. Make sure it's running:\n\n```\nollama serve\n```",
            ""
        )
    except Exception as e:
        return f"❌ Error: {e}", ""

    trace = format_steps(response.steps)
    return response.answer, trace


# ── Example queries ───────────────────────────
EXAMPLES = [
    ["What is the square root of 1764?", DEFAULT_MODEL],
    ["Write Python to generate the first 10 Fibonacci numbers and print them.", DEFAULT_MODEL],
    ["Who founded OpenAI?", DEFAULT_MODEL],
    ["Calculate (2^10) * 3 + sqrt(225)", DEFAULT_MODEL],
]

# ── UI layout ────────────────────────────────
with gr.Blocks(title="Multi-Tool Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
# 🤖 Multi-Tool Agent
An LLM that decides which tool to call — **web search**, **calculator**, or **Python runner**.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            query_box = gr.Textbox(
                label="Your query",
                placeholder="e.g. What is 2^32? / Who invented Python? / Write code to sort a list",
                lines=2,
            )
        with gr.Column(scale=1):
            model_box = gr.Textbox(
                label="Ollama model",
                value=DEFAULT_MODEL,
                placeholder="llama3.2",
            )

    submit_btn = gr.Button("Ask", variant="primary")

    with gr.Row():
        answer_box = gr.Textbox(label="Answer", lines=6, interactive=False)

    with gr.Accordion("Step trace (tool calls made)", open=False):
        trace_box = gr.Markdown()

    gr.Examples(
        examples=EXAMPLES,
        inputs=[query_box, model_box],
        label="Try these",
    )

    submit_btn.click(
        fn=run_query,
        inputs=[query_box, model_box],
        outputs=[answer_box, trace_box],
    )
    query_box.submit(
        fn=run_query,
        inputs=[query_box, model_box],
        outputs=[answer_box, trace_box],
    )

if __name__ == "__main__":
    demo.launch(server_port=7860)