"""エージェントファクトリ (コア)。ツール追加・MCP 追加でこのファイルの編集は不要。"""

from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from app.agent.context import AgentContext
from app.agent.prompts import SYSTEM_PROMPT
from app.core.config import Settings


def build_agent(
    *,
    settings: Settings,
    native_tools: list[BaseTool],
    langmem_tools: list[BaseTool],
    mcp_tools: list[BaseTool],
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
):
    model = ChatOllama(
        model=settings.chat_model,
        base_url=settings.ollama_base_url,
        num_ctx=settings.num_ctx,
        reasoning=settings.reasoning_effort,  # gpt-oss: "low" / "medium" / "high"
        temperature=0,
        validate_model_on_init=True,
    )
    all_tools = [*native_tools, *langmem_tools, *mcp_tools]
    return create_agent(
        model=model,
        tools=all_tools,
        system_prompt=SYSTEM_PROMPT,
        context_schema=AgentContext,
        checkpointer=checkpointer,
        store=store,
    )
