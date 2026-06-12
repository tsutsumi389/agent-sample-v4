"""LangMem hot-path ツール (エージェントが会話中に記憶を保存/検索)。"""

from langchain_core.tools import BaseTool
from langgraph.store.base import BaseStore
from langmem import create_manage_memory_tool, create_search_memory_tool

# namespace は manager.py / store_query.py とバイト単位で一致させること (langmem#140)
MEMORY_NAMESPACE = ("memories", "{langgraph_user_id}")


def langmem_hotpath_tools(store: BaseStore) -> list[BaseTool]:
    return [
        create_manage_memory_tool(namespace=MEMORY_NAMESPACE, store=store),
        create_search_memory_tool(namespace=MEMORY_NAMESPACE, store=store),
    ]
