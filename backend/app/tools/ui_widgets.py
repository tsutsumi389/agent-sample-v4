"""チャット内に対話的UIを描く生成的UIツール群 (宣言的セーフコンポーネント方式)。

`app/tools/` にこのファイルが在るだけで registry が自動登録する (コア無改修)。
各ツールは `@tool(response_format="content_and_artifact")` で
`(要約テキスト, UI封筒)` を返す。封筒は ToolMessage.artifact に格納され、
streaming.py が `ui_resource` SSE イベントとしてフロントへ流す。

封筒の `component` 名はフロントの REGISTRY (table/chart/card/form) と一対一対応する。
props は各 View 側で Zod 検証されてから描画される。
"""

from langchain_core.tools import tool

from app.agent.ui import ui


@tool(response_format="content_and_artifact")
def render_table(title: str, columns: list[str], rows: list[list[str]]):
    """表形式のデータをチャット内に対話的なテーブルとして表示する。

    columns は列見出しの配列、rows は各行 (columns と同じ長さの文字列配列) の配列。
    比較表・一覧・集計結果など、構造化された表データを見せたいときに使う。
    """
    return ui(
        "table",
        summary=f"テーブル「{title}」を表示しました ({len(rows)}行 × {len(columns)}列)。",
        title=title,
        columns=columns,
        rows=rows,
    )


@tool(response_format="content_and_artifact")
def render_bar_chart(title: str, labels: list[str], values: list[float], unit: str = ""):
    """数値データをチャット内に棒グラフとして表示する。

    labels は各棒のラベル、values は対応する数値 (labels と同じ長さ)。
    unit は任意の単位 (例 "件" "%" "円")。推移や比較を可視化したいときに使う。
    """
    return ui(
        "chart",
        summary=f"棒グラフ「{title}」を表示しました ({len(values)}系列)。",
        title=title,
        labels=labels,
        values=values,
        unit=unit,
    )


@tool(response_format="content_and_artifact")
def render_card(title: str, body: str, fields: list[dict]):
    """要点をまとめたカード(見出し+本文+キーバリュー)を表示する。

    fields は [{"label": "...", "value": "..."}] 形式の補足情報の配列
    (補足が無ければ空配列 [] を渡す)。プロフィール・サマリ・結果の要約など、
    1件の情報をきれいに見せたいときに使う。
    """
    return ui(
        "card",
        summary=f"カード「{title}」を表示しました。",
        title=title,
        body=body,
        fields=fields,
    )


@tool(response_format="content_and_artifact")
def render_form(title: str, fields: list[dict], submit_label: str = "送信"):
    """ユーザーに入力してもらうフォームをチャット内に表示する。

    fields は [{"name": "...", "label": "...", "type": "text|number|textarea", "placeholder": "..."}]
    形式の配列。ユーザーがフォームを送信すると、入力内容が次のメッセージとして
    エージェントへ環流する (UI→エージェントの往復)。追加情報を構造的に集めたいときに使う。
    """
    return ui(
        "form",
        summary=f"フォーム「{title}」を表示しました (送信を待っています)。",
        title=title,
        fields=fields,
        submit_label=submit_label,
    )
