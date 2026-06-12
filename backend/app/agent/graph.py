"""エージェントファクトリ (コア)。ツール追加・MCP 追加でこのファイルの編集は不要。

マルチエージェント構成:

    START → orchestrator ─┬→ responder → END                          (高速パス)
                          └→ planner → executor → evaluator ─┬→ executor    (retry / 次ステップ)
                                                              ├→ planner     (replan)
                                                              └→ synthesizer → END

ルーティング (routing.py) は LLM を呼ばない純関数で、各ノードは失敗時の
決定論フォールバックを持つため、LLM の出力がどう壊れても有限ステップで END に到達する。
"""

from collections.abc import Callable
from functools import partial

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore

from app.agent.context import AgentContext
from app.agent.models import default_model_factory
from app.agent.nodes import (
    make_evaluator_node,
    make_executor_node,
    make_orchestrator_node,
    make_planner_node,
    make_synthesizer_node,
)
from app.agent.prompts import EXECUTOR_PROMPT, SYSTEM_PROMPT
from app.agent.routing import Limits, route_after_evaluation, route_after_orchestrator
from app.agent.state import AgentGraphState
from app.core.config import Settings

ModelFactory = Callable[[str, Settings, list[str]], BaseChatModel]


def build_agent(
    *,
    settings: Settings,
    native_tools: list[BaseTool],
    langmem_tools: list[BaseTool],
    mcp_tools: list[BaseTool],
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
    model_factory: ModelFactory | None = None,
):
    make = model_factory or default_model_factory
    # nostream タグは発生源でのトークン配信抑制 (内部思考をチャット欄に混ぜない)
    chat_model = make("chat", settings, [])  # responder: トークン表出
    control_model = make("control", settings, ["nostream"])  # orchestrator/planner/evaluator
    synth_model = make("chat", settings, [])  # synthesizer: トークン表出
    exec_model = make("chat", settings, ["nostream"])  # executor: tool_call/result のみ表出
    all_tools = [*native_tools, *langmem_tools, *mcp_tools]

    # 高速パス: 単一エージェント時代と同一の ReAct をノード直付け
    # (messages チャネルを親と共有し、checkpointer は親から継承)
    responder_agent = create_agent(
        model=chat_model,
        tools=all_tools,
        system_prompt=SYSTEM_PROMPT,
        context_schema=AgentContext,
        store=store,
        name="responder",
    )
    # executor: ステップ毎スクラッチパッド実行 (親 messages 非共有・永続化しない)
    executor_agent = create_agent(
        model=exec_model,
        tools=all_tools,
        system_prompt=EXECUTOR_PROMPT,
        context_schema=AgentContext,
        checkpointer=False,
        store=store,
        name="executor",
    )

    g = StateGraph(AgentGraphState, context_schema=AgentContext)
    g.add_node("orchestrator", make_orchestrator_node(control_model, settings))
    g.add_node("responder", responder_agent)
    g.add_node("planner", make_planner_node(control_model, all_tools, settings))
    g.add_node("executor", make_executor_node(executor_agent, settings))
    g.add_node("evaluator", make_evaluator_node(control_model, settings))
    g.add_node("synthesizer", make_synthesizer_node(synth_model, settings))

    g.add_edge(START, "orchestrator")
    g.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {"responder": "responder", "planner": "planner"},
    )
    g.add_edge("responder", END)
    g.add_edge("planner", "executor")
    g.add_edge("executor", "evaluator")
    g.add_conditional_edges(
        "evaluator",
        partial(route_after_evaluation, limits=Limits.from_settings(settings)),
        {"executor": "executor", "planner": "planner", "synthesizer": "synthesizer"},
    )
    g.add_edge("synthesizer", END)
    return g.compile(checkpointer=checkpointer, store=store, name="agent")
