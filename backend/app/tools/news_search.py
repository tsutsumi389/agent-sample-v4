"""サンプルツール: ニュース検索 / ニュース詳細取得 (仮データ、外部依存なし)。

content_and_artifact 形式: 各ツールは (LLM向けテキスト, 構造化データ) の2要素を返す。
テキストは ReAct ループ内で LLM が読む要約、artifact は ToolMessage.artifact に
そのまま載る生データで、executor 側が PlanStep.data へ回収する用途を想定する。
"""

from langchain_core.tools import tool

from app.agent.ui import ui

# 仮のニュース記事データベース (id をキーに詳細本文まで保持)。
# 実運用では外部ニュースAPIの戻り値に置き換える前提のスタブ。
_ARTICLES: list[dict] = [
    {
        "id": "n001",
        "title": "国産大規模言語モデル、推論コストを半減する新手法を発表",
        "summary": "研究チームが投機的デコーディングを応用し、推論コストを約50%削減したと報告。",
        "source": "Tech Daily",
        "published": "2026-06-17",
        "category": "technology",
        "url": "https://news.example.com/n001",
        "body": (
            "国内の研究チームは6月17日、大規模言語モデルの推論コストを従来比で約50%"
            "削減する新手法を発表した。投機的デコーディングと量子化を組み合わせ、"
            "精度を維持したまま処理速度を2.1倍に高めたという。年内のOSS公開を予定。"
        ),
    },
    {
        "id": "n002",
        "title": "中央銀行、政策金利を据え置き 物価見通しは上方修正",
        "summary": "市場予想どおり金利は据え置き。一方でインフレ見通しは小幅に引き上げられた。",
        "source": "Economy Wire",
        "published": "2026-06-16",
        "category": "economy",
        "url": "https://news.example.com/n002",
        "body": (
            "中央銀行は6月16日の会合で政策金利の据え置きを決定した。声明では、"
            "賃金上昇を背景に物価見通しを小幅に上方修正。次回会合での利上げ観測が"
            "市場で強まり、長期金利は一時的に上昇した。"
        ),
    },
    {
        "id": "n003",
        "title": "再生可能エネルギー比率、過去最高の42%に到達",
        "summary": "前年の38%から上昇。太陽光と洋上風力の新設が寄与した。",
        "source": "Green Report",
        "published": "2026-06-15",
        "category": "environment",
        "url": "https://news.example.com/n003",
        "body": (
            "今年度の発電量に占める再生可能エネルギーの比率が42%に達し、過去最高を"
            "更新した。洋上風力の大型案件が相次いで稼働したことが主因で、政府は2030年"
            "目標の前倒し達成も視野に入るとしている。"
        ),
    },
    {
        "id": "n004",
        "title": "新型ロケット、商業初打ち上げに成功 小型衛星12基を投入",
        "summary": "民間企業の新型機が初の商業打ち上げに成功。再使用機の量産に弾み。",
        "source": "Space Now",
        "published": "2026-06-14",
        "category": "technology",
        "url": "https://news.example.com/n004",
        "body": (
            "民間宇宙企業の新型ロケットが6月14日、商業初打ち上げに成功し、小型衛星"
            "12基を予定軌道へ投入した。第1段は洋上の回収船に着地し再使用に成功。"
            "同社は今後、月1回ペースでの打ち上げを目指すとしている。"
        ),
    },
    {
        "id": "n005",
        "title": "国際サッカー親善試合、代表が逆転勝利でW杯へ弾み",
        "summary": "後半に2点を返し2-1で勝利。新戦術が機能したと監督。",
        "source": "Sports Press",
        "published": "2026-06-13",
        "category": "sports",
        "url": "https://news.example.com/n005",
        "body": (
            "国際親善試合で代表は先制を許したものの、後半に2得点を挙げて2-1で逆転"
            "勝利した。監督は新たに導入した可変式の布陣が機能したと評価。来月の"
            "ワールドカップ本大会に向け好材料となった。"
        ),
    },
]

_BY_ID = {a["id"]: a for a in _ARTICLES}

# 検索結果には本文(body)を含めない軽量ビューを返す (詳細は news_detail で取得)。
_LIST_FIELDS = ("id", "title", "summary", "source", "published", "category", "url")


def _to_list_view(article: dict) -> dict:
    return {k: article[k] for k in _LIST_FIELDS}


@tool(response_format="content_and_artifact")
def news_search(query: str, limit: int = 5) -> tuple[str, list[dict]]:
    """キーワードでニュースを検索し、見出しと要約の一覧を返す (本文は含めない)。

    詳細な本文が必要な場合は、結果の id を news_detail に渡して取得する。
    """
    q = (query or "").strip().lower()
    if q:
        hits = [
            a
            for a in _ARTICLES
            if q in a["title"].lower()
            or q in a["summary"].lower()
            or q in a["category"].lower()
        ]
    else:
        hits = list(_ARTICLES)
    # 新しい順 (published 降順) に整列してから件数制限。
    hits.sort(key=lambda a: a["published"], reverse=True)
    hits = hits[: max(1, limit)]

    artifact = [_to_list_view(a) for a in hits]
    if not artifact:
        return f"「{query}」に一致するニュースは見つかりませんでした。", []

    lines = [f"「{query}」の検索結果 {len(artifact)}件:"]
    lines += [
        f"- [{a['id']}] {a['title']} ({a['source']} / {a['published']})\n    {a['summary']}"
        for a in artifact
    ]
    return "\n".join(lines), artifact


@tool(response_format="content_and_artifact")
def news_detail(article_id: str) -> tuple[str, dict]:
    """ニュース記事IDを指定して、本文を含む詳細を取得する (news_search の id を使う)。"""
    article = _BY_ID.get((article_id or "").strip())
    if article is None:
        known = ", ".join(_BY_ID)
        return f"記事ID「{article_id}」は見つかりませんでした。利用可能なID: {known}", {}

    text = (
        f"{article['title']}\n"
        f"出典: {article['source']} / {article['published']} / {article['category']}\n"
        f"URL: {article['url']}\n\n"
        f"{article['body']}"
    )
    return text, dict(article)


# 表示に使うフィールドだけを通す (artifact 肥大とコンテキスト汚染を避ける)。
_NEWS_VIEW_FIELDS = ("id", "title", "summary", "source", "published", "category", "url", "body")


@tool(response_format="content_and_artifact")
def render_news(title: str, articles: list[dict]):
    """ニュース記事の一覧をチャット内に専用UI(ニュースカード)として表示する。

    articles は news_search / news_detail が返した記事オブジェクトの配列。
    各記事は title を必須とし、summary/source/published/category/url/body は任意
    (body があれば本文付きの詳細表示になる)。検索結果や記事詳細をユーザーに
    見やすく提示したいときに使う。
    """
    # LLM が余計なキーを混ぜても表示用フィールドだけに正規化する (フロントの Zod と二重防御)。
    cleaned = [
        {k: a[k] for k in _NEWS_VIEW_FIELDS if isinstance(a, dict) and a.get(k) is not None}
        for a in articles
    ]
    return ui(
        "news",
        summary=f"ニュース「{title}」を表示しました ({len(cleaned)}件)。",
        title=title,
        articles=cleaned,
    )
