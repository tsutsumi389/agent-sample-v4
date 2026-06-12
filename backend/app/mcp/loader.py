"""MCP ローダー。mcp_servers.json (Claude Desktop 互換) を読むだけでサーバが増える。"""

import json
import logging
import os
import re
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _substitute_env(value):
    if isinstance(value, str):
        return _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, list):
        return [_substitute_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    return value


def load_mcp_config(path: Path) -> dict:
    """mcpServers を読み込み、${ENV} を置換し、transport を補完して返す。"""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    servers = _substitute_env(raw.get("mcpServers", {}))
    for name, conf in servers.items():
        if "transport" not in conf:
            if "command" in conf:
                conf["transport"] = "stdio"
            elif "url" in conf:
                conf["transport"] = "streamable_http"
            else:
                raise ValueError(f"MCP サーバ '{name}' に command も url もありません")
    return servers


def build_mcp_client(config: dict) -> MultiServerMCPClient | None:
    if not config:
        return None
    return MultiServerMCPClient(config, tool_name_prefix=True)


async def get_mcp_tools(client: MultiServerMCPClient | None) -> list[BaseTool]:
    if client is None:
        return []
    return await client.get_tools()
