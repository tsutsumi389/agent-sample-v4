"""意味記憶: 構造化ユーザープロファイル (Profile パターン, 単一ドキュメント)。

エピソード記憶 (自由テキスト Collection, ("memories", user_id)) とは分け、安定した
ユーザー属性 (名前・職業・恒常的な好み・制約・文体) だけを 1 ドキュメントに保持する。
ターン開始時に丸ごとロードして responder / planner / synthesizer のシステムプロンプト
へ注入することで、search_memory に頼らず初手からパーソナライズする。
"""

from typing import Any

from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

# namespace は tools.py の MEMORY_NAMESPACE と同じ規約 (LangMem manager 用テンプレート)。
PROFILE_NAMESPACE = ("profile", "{langgraph_user_id}")
# Profile パターンの単一ドキュメントキー。
PROFILE_KEY = "default"


class UserProfile(BaseModel):
    """ユーザーの安定した属性 (意味記憶)。一時的な出来事・作業ログは含めない。"""

    name: str | None = Field(default=None, description="ユーザーの名前または呼ばれ方")
    occupation: str | None = Field(default=None, description="職業・役割・専門分野")
    communication_style: str | None = Field(
        default=None,
        description="好む回答の文体・口調・長さ (例: 簡潔に、丁寧に、専門的に)",
    )
    preferences: list[str] = Field(
        default_factory=list,
        description="恒常的な好み・嗜好 (その場限りの話題は含めない)",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="守るべき制約・避けたいこと (例: カフェインを避ける、予算が限られる)",
    )


# スカラー項目のラベル (表示順を兼ねる)。リスト項目は format_profile で個別に扱う。
_SCALAR_LABELS = {
    "name": "名前",
    "occupation": "職業",
    "communication_style": "希望する文体",
}


def _coerce_profile(value: Any) -> UserProfile | None:
    """store の値形状 ({"kind":..., "content": {...}} / 直 dict) を UserProfile へ。"""
    data = value
    if isinstance(data, dict) and "content" in data:
        data = data["content"]
    if not isinstance(data, dict):
        return None
    try:
        return UserProfile.model_validate(data)
    except Exception:
        return None


def format_profile(value: Any) -> str:
    """UserProfile を空フィールドを省いた人間可読テキストへ整形する。全空なら ""。"""
    profile = _coerce_profile(value)
    if profile is None:
        return ""
    lines: list[str] = []
    for field_name, label in _SCALAR_LABELS.items():
        v = getattr(profile, field_name)
        if v:
            lines.append(f"- {label}: {v}")
    if profile.preferences:
        lines.append("- 好み: " + "、".join(profile.preferences))
    if profile.constraints:
        lines.append("- 制約: " + "、".join(profile.constraints))
    return "\n".join(lines)


async def get_profile_text(store: BaseStore, user_id: str) -> str:
    """意味記憶 (ユーザープロファイル) をロードして整形テキストで返す。

    best-effort: 取得失敗・未登録時は "" を返し、回答生成を止めない。
    """
    try:
        item = await store.aget(("profile", user_id), PROFILE_KEY)
    except Exception:
        return ""
    if item is None:
        return ""
    return format_profile(item.value)
