import pytest
from langchain_core.tools import tool

from app.tools.registry import ToolRegistry, build_registry


def test_discovers_sample_tools():
    registry = build_registry()
    names = {t.name for t in registry.all()}
    assert {"current_datetime", "calculate", "web_search"} <= names


def test_describe_includes_source():
    registry = build_registry()
    info = registry.describe()
    assert all(d["source"] == "native" for d in info)
    assert all(d["name"] and d["description"] for d in info)


def test_duplicate_name_raises():
    @tool
    def sample_tool() -> str:
        """サンプル。"""
        return "x"

    registry = ToolRegistry()
    registry.register(sample_tool)
    with pytest.raises(ValueError, match="duplicate tool name"):
        registry.register(sample_tool)
