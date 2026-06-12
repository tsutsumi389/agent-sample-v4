"""サンプルツール: 現在日時 (JST)。"""

from datetime import datetime, timedelta, timezone

from langchain_core.tools import tool

JST = timezone(timedelta(hours=9))


@tool
def current_datetime() -> str:
    """現在の日時 (日本時間 JST) を返す。"""
    now = datetime.now(JST)
    weekdays = ["月", "火", "水", "木", "金", "土", "日"]
    return f"{now.strftime('%Y-%m-%d %H:%M:%S')} JST ({weekdays[now.weekday()]}曜日)"
