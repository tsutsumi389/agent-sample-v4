"""サンプルツール: 安全な数式評価。"""

import ast
import operator

from langchain_core.tools import tool

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError(f"未対応の式です: {ast.dump(node)}")


@tool
def calculate(expression: str) -> str:
    """数式を計算して結果を返す。四則演算・べき乗・剰余に対応 (例: "(2+3)*4**2")。"""
    try:
        result = _eval(ast.parse(expression, mode="eval"))
        return str(result)
    except Exception as exc:
        return f"計算エラー: {exc}"
