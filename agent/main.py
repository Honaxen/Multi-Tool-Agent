"""
main.py — CLI interface for the multi-tool agent.

Usage:
    python3 -m agent.main
    python3 -m agent.main --model llama3.2
    python3 -m agent.main --query "What is 2^32?"
"""

import argparse
import sys
from agent.agent import run_agent, DEFAULT_MODEL

# ── ANSI colors ──────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
PURPLE = "\033[35m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"


def print_step(step, index: int):
    if step.type == "tool_call":
        print(f"\n{CYAN}{BOLD}[Step {index}] Tool call → {step.tool_name or '?'}{RESET}")
        if step.tool_args:
            print(f"{DIM}  args: {step.tool_args}{RESET}")

    elif step.type == "tool_result":
        print(f"{YELLOW}[Step {index}] Tool result ({step.tool_name}){RESET}")
        # Truncate long results for display
        result = step.content
        if len(result) > 400:
            result = result[:400] + f"\n{DIM}  ... (truncated){RESET}"
        for line in result.splitlines():
            print(f"  {DIM}{line}{RESET}")

    elif step.type == "answer":
        print(f"\n{GREEN}{BOLD}[Answer]{RESET}")
        print(step.content)


def interactive_loop(model: str):
    print(f"{BOLD}Multi-Tool Agent{RESET} — model: {PURPLE}{model}{RESET}")
    print(f"{DIM}Tools: web_search, calculate, run_python{RESET}")
    print(f"{DIM}Type 'quit' or Ctrl+C to exit.\n{RESET}")

    while True:
        try:
            query = input(f"{BOLD}You:{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        print(f"\n{DIM}Thinking...{RESET}")
        try:
            response = run_agent(query, model=model)
        except ConnectionError as e:
            print(f"{RED}Connection error:{RESET} {e}")
            continue
        except Exception as e:
            print(f"{RED}Error:{RESET} {e}")
            continue

        for i, step in enumerate(response.steps, 1):
            print_step(step, i)

        print()  # spacing before next prompt


def main():
    parser = argparse.ArgumentParser(description="Multi-tool agent CLI")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--query", default=None, help="Single query (non-interactive)")
    args = parser.parse_args()

    if args.query:
        # Single-shot mode
        try:
            response = run_agent(args.query, model=args.model)
        except ConnectionError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        for i, step in enumerate(response.steps, 1):
            print_step(step, i)
    else:
        interactive_loop(model=args.model)


if __name__ == "__main__":
    main()