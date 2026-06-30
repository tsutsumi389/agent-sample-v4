"""ノード共通ヘルパー。"""

import json
import logging
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer

from app.agent.parsing import (
    DataSelectionSchema,
    EntrySelectionSchema,
    content_to_text,
    structured_or_parse,
)
from app.agent.prompts import SCREENING_SYSTEM, screening_user
from app.core.config import Settings

logger = logging.getLogger(__name__)

# executor が例外・反復上限で打ち切られたことを evaluator に伝えるマーカー
EXECUTION_FAILED_MARKER = "(実行打ち切り: "


def safe_stream_writer() -> Callable[[Any], None]:
    """get_stream_writer はランタイム外で例外を投げるため、ユニットテストでも安全な no-op を返す。"""
    try:
        return get_stream_writer()
    except Exception:
        return lambda _payload: None


def last_human_text(state: dict) -> str:
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, HumanMessage):
            return content_to_text(msg.content)
    return ""


def sanitize_dependencies(plan: list[dict]) -> None:
    """plan の depends_on を破壊的に正規化する (planner 出力の堅牢化)。

    - 自己参照・重複・存在しない id 参照を除去。
    - 循環があれば Kahn 法で検出し、循環に残ったノードの依存をクリアして前進可能にする
      (循環は永遠に ready にならず停止性は routing で保たれるが、結果が出ないため壊す)。
    """
    by_id = {s["id"]: s for s in plan}
    for s in plan:
        s["depends_on"] = [
            d for d in dict.fromkeys(s.get("depends_on") or []) if d in by_id and d != s["id"]
        ]
    indeg = {sid: len(s["depends_on"]) for sid, s in by_id.items()}
    nexts: dict[int, list[int]] = {sid: [] for sid in by_id}
    for s in plan:
        for d in s["depends_on"]:
            nexts[d].append(s["id"])
    queue = [sid for sid, deg in indeg.items() if deg == 0]
    seen = 0
    while queue:
        sid = queue.pop()
        seen += 1
        for nx in nexts[sid]:
            indeg[nx] -= 1
            if indeg[nx] == 0:
                queue.append(nx)
    if seen < len(by_id):
        # 循環あり: indeg>0 のノード (循環本体＋それに間接依存する下流ノード) の依存を
        # 安全側で全クリアし、全ステップを即時実行可能にする。下流の非循環依存も巻き込まれ
        # 順序保証は失うが、前進性 (必ず ready になる) を優先する。発火は planner(LLM) が
        # 循環を出した異常時のみ。
        for sid, deg in indeg.items():
            if deg > 0:
                by_id[sid]["depends_on"] = []


def format_step_data(data: list[dict] | None) -> str:
    """PlanStep.data (artifact 群) を LLM プロンプト用の JSON テキストへ整形する (全量・無切詰め)。

    回収不能・空のときは空文字。構造化データは機械処理用 (id・数値等) のため、result(テキスト要約)
    と違い途中で切り詰めない — truncate すると JSON が壊れたり値が欠落して不正確になるため。
    各ツール由来を後段で判別できるよう {"tool", "artifact"} のまま並べる。
    """
    if not data:
        return ""
    try:
        return json.dumps(
            [{"tool": d.get("tool"), "artifact": d.get("artifact")} for d in data],
            ensure_ascii=False,
            default=str,
        )
    except Exception:
        return ""


def _project_dict(d: Any, keep_fields: list[str]) -> Any:
    """dict を keep_fields のキーだけに射影する (keep_fields 空なら全フィールド)。"""
    if not isinstance(d, dict) or not keep_fields:
        return d
    return {k: d[k] for k in keep_fields if k in d}


def _project_artifact(artifact: Any, keep_fields: list[str], keep_items: list[int] | None) -> Any:
    """1 ツールの artifact を選択指定で射影する。値は元のまま (生成しない)。"""
    if isinstance(artifact, list):
        items = artifact
        if keep_items:
            items = [items[j] for j in keep_items if isinstance(j, int) and 0 <= j < len(items)]
        return [_project_dict(it, keep_fields) for it in items]
    if isinstance(artifact, dict):
        return _project_dict(artifact, keep_fields)
    return artifact  # スカラ等はそのまま


def apply_data_selection(data: list[dict], selection: DataSelectionSchema) -> list[dict]:
    """選択指定 (LLM 出力) を元 data に決定論的に適用し、必要箇所だけに絞った data を返す。

    LLM は値を生成しない (フィールド名・項目位置の指定のみ)。射影は元データから抜き出すだけなので
    残った値は元のまま。安全側として、選択が空・結果が空のときは元 data を全量返す
    (誤って全ドロップしない / structured_or_parse の fallback=空選択もここで全量に化ける)。
    """
    if not data or not selection.selections:
        return data
    by_index: dict[int, EntrySelectionSchema] = {s.index: s for s in selection.selections}
    out: list[dict] = []
    for i, entry in enumerate(data):
        sel = by_index.get(i)
        if sel is None:
            continue  # 選択に挙がらなかったエントリは捨てる
        out.append(
            {
                "tool": entry.get("tool"),
                "artifact": _project_artifact(entry.get("artifact"), sel.keep_fields, sel.keep_items),
            }
        )
    return out or data  # 何も残らなければ全量フォールバック


async def screen_step_data(
    model: Any, data: list[dict] | None, purpose: str, settings: Settings
) -> list[dict] | None:
    """data を purpose に必要な箇所だけへ LLM スクリーニングする (値は変えず選択のみ)。

    data が空なら LLM を呼ばずそのまま返す。選択取得や適用に失敗しても全量へフォールバックし、
    例外は投げない契約 (executor/evaluator の「例外を漏らさない」並列実行と整合)。
    """
    if not data:
        return data
    try:
        messages = [
            SystemMessage(content=SCREENING_SYSTEM),
            HumanMessage(content=screening_user(purpose, format_step_data(data))),
        ]
        selection = await structured_or_parse(
            model,
            messages,
            DataSelectionSchema,
            use_structured=settings.supports_structured_output,
            fallback=DataSelectionSchema(selections=[]),
        )
        return apply_data_selection(data, selection)
    except Exception:
        logger.exception("構造化データのスクリーニングに失敗したため全量を渡します")
        return data


def format_feedback_history(history: list[str]) -> str:
    """retry 毎の評価者指摘の履歴を、古い順に番号付きで1本のテキストへ整形する。

    executor (全指摘の改善) と evaluator (全指摘の反映度評価) の両方が同じ体裁で読む。
    末尾の番号が最新の指摘。空履歴なら "" を返す。"""
    return "\n".join(f"{i}. {fb}" for i, fb in enumerate(history, 1) if fb)


def ready_step_indices(plan: list[dict]) -> list[int]:
    """依存 (depends_on) が全て done で、まだ実行待ち (pending) のステップの index 群。"""
    done = {s["id"] for s in plan if s.get("status") == "done"}
    return [
        i
        for i, s in enumerate(plan)
        if s.get("status") == "pending" and all(d in done for d in (s.get("depends_on") or []))
    ]
