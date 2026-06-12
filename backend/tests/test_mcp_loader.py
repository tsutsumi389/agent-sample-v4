import json

import pytest

from app.mcp.loader import build_mcp_client, load_mcp_config


def _write(tmp_path, data):
    path = tmp_path / "mcp_servers.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_empty_config_returns_none(tmp_path):
    path = _write(tmp_path, {"mcpServers": {}})
    assert load_mcp_config(path) == {}
    assert build_mcp_client({}) is None


def test_missing_file_returns_empty(tmp_path):
    assert load_mcp_config(tmp_path / "nope.json") == {}


def test_env_substitution_and_transport_inference(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "secret123")
    path = _write(
        tmp_path,
        {
            "mcpServers": {
                "stdio_server": {
                    "command": "uvx",
                    "args": ["some-mcp"],
                    "env": {"TOKEN": "${MY_TOKEN}"},
                },
                "http_server": {"url": "http://localhost:9000/mcp"},
            }
        },
    )
    config = load_mcp_config(path)
    assert config["stdio_server"]["transport"] == "stdio"
    assert config["stdio_server"]["env"]["TOKEN"] == "secret123"
    assert config["http_server"]["transport"] == "streamable_http"


def test_invalid_server_raises(tmp_path):
    path = _write(tmp_path, {"mcpServers": {"broken": {"foo": "bar"}}})
    with pytest.raises(ValueError, match="broken"):
        load_mcp_config(path)
