"""
tests/test_tools.py — Unit tests for all 3 tools and the tool executor.

Run:
    pytest tests/ -v
"""

import pytest
from agent.tools import web_search, calculate, run_python
from agent.tool_executor import parse_tool_call, execute_tool, ToolCallError


# ─────────────────────────────────────────────
# Calculator tests
# ─────────────────────────────────────────────

class TestCalculator:
    def test_basic_addition(self):
        result = calculate("2 + 3")
        assert "5" in result

    def test_basic_multiplication(self):
        result = calculate("6 * 7")
        assert "42" in result

    def test_power(self):
        result = calculate("2 ** 10")
        assert "1024" in result

    def test_sqrt(self):
        result = calculate("sqrt(144)")
        assert "12.0" in result

    def test_complex_expression(self):
        result = calculate("(10 + 5) * 2 - 3")
        assert "27" in result

    def test_division_by_zero(self):
        result = calculate("1 / 0")
        assert "zero" in result.lower()

    def test_disallowed_name(self):
        result = calculate("__import__('os')")
        assert "error" in result.lower() or "not allowed" in result.lower()


# ─────────────────────────────────────────────
# Python runner tests
# ─────────────────────────────────────────────

class TestRunPython:
    def test_hello_world(self):
        result = run_python('print("hello world")')
        assert "hello world" in result

    def test_list_generation(self):
        result = run_python("print([x**2 for x in range(5)])")
        assert "[0, 1, 4, 9, 16]" in result

    def test_fibonacci(self):
        code = """
a, b = 0, 1
fibs = []
for _ in range(7):
    fibs.append(a)
    a, b = b, a + b
print(fibs)
"""
        result = run_python(code)
        assert "0" in result and "1" in result and "13" in result

    def test_syntax_error(self):
        result = run_python("print(")
        assert "error" in result.lower() or "SyntaxError" in result

    def test_blocked_import(self):
        result = run_python("import os; os.system('ls')")
        # os.system is blocked
        assert "blocked" in result.lower() or "error" in result.lower()

    def test_no_output(self):
        result = run_python("x = 42")
        assert "no output" in result.lower()


# ─────────────────────────────────────────────
# Tool executor tests
# ─────────────────────────────────────────────

class TestToolExecutor:
    def test_parse_valid_json(self):
        raw = '{"tool": "calculate", "args": {"expression": "2 + 2"}}'
        name, args = parse_tool_call(raw)
        assert name == "calculate"
        assert args["expression"] == "2 + 2"

    def test_parse_with_code_fence(self):
        raw = '```json\n{"tool": "run_python", "args": {"code": "print(1)"}}\n```'
        name, args = parse_tool_call(raw)
        assert name == "run_python"

    def test_parse_missing_tool_key(self):
        with pytest.raises(ToolCallError):
            parse_tool_call('{"args": {"query": "test"}}')

    def test_parse_unknown_tool(self):
        with pytest.raises(ToolCallError):
            parse_tool_call('{"tool": "nonexistent", "args": {}}')

    def test_execute_calculate(self):
        result = execute_tool("calculate", {"expression": "3 * 3"})
        assert "9" in str(result)

    def test_execute_run_python(self):
        result = execute_tool("run_python", {"code": 'print("tool works")'})
        assert "tool works" in result

    def test_execute_bad_args(self):
        with pytest.raises(ToolCallError):
            execute_tool("calculate", {"wrong_arg": "oops"})


# ─────────────────────────────────────────────
# Web search test (network — skip if offline)
# ─────────────────────────────────────────────

class TestWebSearch:
    def test_returns_string(self):
        result = web_search("Python programming language")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_unknown_query_graceful(self):
        result = web_search("xkqzjvmwpqrstuvwxyz1234567890")
        assert isinstance(result, str)
