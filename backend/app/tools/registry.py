"""ネイティブツールの自動探索レジストリ。

`app/tools/` にモジュールを置き、module-level の `@tool` オブジェクトを定義するだけで
エージェントに登録される。コア (graph.py 等) の変更は不要。
さらに entry_points(group="agent.tools") でサードパーティプラグインも探索する。
"""

import importlib
import logging
import pkgutil
from importlib.metadata import entry_points

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

_EXCLUDED_MODULES = {"registry", "base", "__init__"}
ENTRY_POINT_GROUP = "agent.tools"


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[BaseTool, str]] = {}

    def register(self, tool: BaseTool, source: str = "native") -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = (tool, source)

    def discover_package(self, package_name: str = "app.tools") -> None:
        package = importlib.import_module(package_name)
        for mod_info in pkgutil.iter_modules(package.__path__):
            if mod_info.name in _EXCLUDED_MODULES:
                continue
            module = importlib.import_module(f"{package_name}.{mod_info.name}")
            for obj in vars(module).values():
                if isinstance(obj, BaseTool):
                    self.register(obj, source="native")

    def discover_entry_points(self, group: str = ENTRY_POINT_GROUP) -> None:
        for ep in entry_points(group=group):
            try:
                obj = ep.load()
            except Exception:
                logger.exception("ツールプラグイン %s のロードに失敗しました", ep.name)
                continue
            tools = obj if isinstance(obj, (list, tuple)) else [obj]
            for tool in tools:
                if isinstance(tool, BaseTool):
                    self.register(tool, source="plugin")

    def all(self) -> list[BaseTool]:
        return [tool for tool, _ in self._tools.values()]

    def describe(self) -> list[dict]:
        return [
            {"name": tool.name, "description": tool.description, "source": source}
            for tool, source in self._tools.values()
        ]


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.discover_package()
    registry.discover_entry_points()
    return registry
