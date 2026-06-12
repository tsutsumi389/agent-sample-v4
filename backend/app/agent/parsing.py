"""LLM 出力の堅牢な JSON パース。

ローカル LLM (gpt-oss / qwen3) は構造化出力 (trustcall / with_structured_output) が
不安定なため使わず、プレーンテキストから JSON を抽出して pydantic で検証する。
全関数は「絶対に例外を投げない」契約 — 失敗は None / fallback で表現する。
"""

import json
import logging
import re
from collections.abc import Callable
from typing import Any, Literal, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

T = TypeVar("T", bound=BaseModel)


class PlanSchema(BaseModel):
    steps: list[str]

    @field_validator("steps", mode="before")
    @classmethod
    def coerce(cls, v: Any) -> list[str]:
        # list[dict] で来ても description/step/task/content キーを拾う。空要素は除去。
        out: list[str] = []
        for item in v if isinstance(v, list) else []:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                for k in ("description", "step", "task", "content"):
                    if isinstance(item.get(k), str) and item[k].strip():
                        out.append(item[k].strip())
                        break
        return out


class VerdictSchema(BaseModel):
    verdict: Literal["pass", "retry", "replan"]
    feedback: str = ""

    @field_validator("feedback", mode="before")
    @classmethod
    def coerce_feedback(cls, v: Any) -> str:
        return v if isinstance(v, str) else ""


def content_to_text(content: Any) -> str:
    """message content (str | list[block]) をプレーンテキストへ。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content) if content is not None else ""


def strip_think(text: str) -> str:
    """qwen3 等の <think>...</think> ブロックを除去する。"""
    return THINK_RE.sub("", text).strip()


def extract_json(text: str) -> dict | None:
    """テキストから最初の valid な JSON オブジェクトを抽出する。

    (a) <think> 除去 → (b) コードフェンス内優先 → (c) 各 '{' 位置から
    raw_decode 走査。全滅で None。
    """
    if not isinstance(text, str) or not text:
        return None
    cleaned = strip_think(text)
    candidates = [m.group(1) for m in FENCE_RE.finditer(cleaned)]
    candidates.append(cleaned)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for match in re.finditer(r"\{", candidate):
            try:
                obj, _ = decoder.raw_decode(candidate[match.start() :])
            except (ValueError, TypeError):
                continue
            if isinstance(obj, dict):
                return obj
    return None


def parse_json_as(text: str, schema: type[T]) -> T | None:
    """extract_json + pydantic 検証。失敗は None。例外は投げない。"""
    obj = extract_json(text)
    if obj is None:
        return None
    try:
        return schema.model_validate(obj)
    except Exception:
        return None


async def parse_with_retry(
    model: Any,
    messages: list[BaseMessage],
    schema: type[T],
    *,
    max_retries: int = 2,
    fallback: T | Callable[[], T],
) -> T:
    """LLM 呼び出し＋パースを最大 max_retries 回試行し、全滅したら必ず fallback を返す。

    パース失敗時は不正出力を踏まえた修正指示を追記して再呼び出しする。
    LLM 自体の例外 (接続断含む) もリトライ消費とする。バックオフなし (ローカル LLM)。
    """
    convo = list(messages)
    for _ in range(max(1, max_retries)):
        try:
            response = await model.ainvoke(convo)
        except Exception:
            logger.exception("LLM 呼び出しに失敗しました (schema=%s)", schema.__name__)
            continue
        text = content_to_text(getattr(response, "content", response))
        parsed = parse_json_as(text, schema)
        if parsed is not None:
            return parsed
        convo = [
            *convo,
            response if isinstance(response, BaseMessage) else HumanMessage(content=text),
            HumanMessage(
                content="前回の出力はJSONとして不正でした。"
                "説明やコードフェンスを付けず、指定した形式のJSONオブジェクトのみを出力してください。"
            ),
        ]
    logger.warning("JSON パースが全滅したため fallback を使用します (schema=%s)", schema.__name__)
    return fallback() if callable(fallback) else fallback
