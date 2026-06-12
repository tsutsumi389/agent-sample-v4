"""サンプルツール: Web検索 (DuckDuckGo HTML、APIキー不要)。"""

import html
import re
import urllib.parse
import urllib.request

from langchain_core.tools import tool

_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]*>(.*?)</a>.*?'
    r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _clean(fragment: str) -> str:
    return html.unescape(_TAG_RE.sub("", fragment)).strip()


@tool
def web_search(query: str) -> str:
    """Webを検索して上位結果のタイトルと概要を返す。"""
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (agent-sample)"})
    try:
        with urllib.request.urlopen(req, timeout=10) as res:
            body = res.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return f"検索に失敗しました: {exc}"
    results = [
        f"- {_clean(title)}: {_clean(snippet)}"
        for title, snippet in _RESULT_RE.findall(body)[:5]
    ]
    if not results:
        return "検索結果が見つかりませんでした。"
    return "\n".join(results)
