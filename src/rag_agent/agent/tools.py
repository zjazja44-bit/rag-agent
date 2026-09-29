"""Tools the agent can call, plus a tiny registry.

Each tool bundles a JSON Schema description (sent to the model) with a Python
callable that produces a string observation. The registry filters incoming
arguments against the callable's signature and never raises into the loop — a
bad tool call becomes an observation the model can recover from.
"""
from __future__ import annotations

import ast
import inspect
import operator
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from ..rag.chain import build_context


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool {name!r}. Available: {self.names()}"
        allowed = inspect.signature(tool.fn).parameters
        kwargs = {k: v for k, v in (arguments or {}).items() if k in allowed}
        try:
            return str(tool.fn(**kwargs))
        except Exception as exc:  # surface the error as an observation
            return f"Error: {type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# calculator
# --------------------------------------------------------------------------- #
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_eval(expression: str) -> float:
    """Evaluate an arithmetic expression without ``eval`` (AST whitelist only)."""
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression element")


def make_calculator_tool() -> Tool:
    def calculator(expression: str) -> str:
        value = safe_eval(expression)
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    return Tool(
        name="calculator",
        description="计算一个算术表达式，支持 + - * / // % ** 和括号。",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "算术表达式，如 (12+3)*4"}
            },
            "required": ["expression"],
        },
        fn=calculator,
    )


def make_knowledge_search_tool(retriever) -> Tool:
    def knowledge_search(query: str, top_k: int = 5) -> str:
        hits = retriever.retrieve(query, top_k=int(top_k))
        if not hits:
            return "（未检索到相关资料）"
        return build_context(hits)

    return Tool(
        name="knowledge_search",
        description="在知识库中检索与问题相关的资料片段，返回带编号的原文。回答事实性问题前应先调用它。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "用于检索的问题或关键词"},
                "top_k": {"type": "integer", "description": "返回片段数量", "default": 5},
            },
            "required": ["query"],
        },
        fn=knowledge_search,
    )


_BUILDERS: dict[str, Callable[[Any], Tool]] = {
    "calculator": lambda retriever: make_calculator_tool(),
    "knowledge_search": lambda retriever: make_knowledge_search_tool(retriever),
}


def build_tools(retriever, names: Sequence[str]) -> ToolRegistry:
    registry = ToolRegistry()
    for name in names:
        builder = _BUILDERS.get(name)
        if builder is None:
            raise ValueError(f"Unknown tool: {name!r}. Known: {list(_BUILDERS)}")
        registry.register(builder(retriever))
    return registry
