"""エージェント実行コンテキスト (create_agent の context_schema)。"""

from dataclasses import dataclass


@dataclass
class AgentContext:
    user_id: str
