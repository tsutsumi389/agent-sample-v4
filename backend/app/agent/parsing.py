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
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

T = TypeVar("T", bound=BaseModel)


def _coerce_opt_int(v: Any) -> int | None:
    """単一の int 値の堅牢化: int/数字文字列のみ拾い、それ以外は None。"""
    if isinstance(v, bool):  # bool は int サブクラスなので明示除外
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.strip().lstrip("-").isdigit():
        return int(v.strip())
    return None


def _coerce_int_list(v: Any) -> list[int]:
    """depends_on の堅牢化: int/数字文字列のみ拾い、それ以外は捨てる。"""
    out: list[int] = []
    for item in v if isinstance(v, list) else []:
        coerced = _coerce_opt_int(item)
        if coerced is not None:
            out.append(coerced)
    return out


class PlanStepSchema(BaseModel):
    # LLM が出した元 id (depends_on はこの id 空間で書かれる)。planner が出現順 1..N へ
    # リマップする際の対応付けに使う。欠落・不正なら None (出現位置で代替)。
    id: int | None = None
    description: str
    instruction: str = ""  # executor 向けの具体的・パーソナライズ済み実行手順 (任意)
    depends_on: list[int] = []

    @field_validator("id", mode="before")
    @classmethod
    def coerce_id(cls, v: Any) -> int | None:
        return _coerce_opt_int(v)

    @field_validator("instruction", mode="before")
    @classmethod
    def coerce_instruction(cls, v: Any) -> str:
        return v.strip() if isinstance(v, str) else ""

    @field_validator("depends_on", mode="before")
    @classmethod
    def coerce_depends_on(cls, v: Any) -> list[int]:
        return _coerce_int_list(v)


class PlanSchema(BaseModel):
    steps: list[PlanStepSchema]

    @field_validator("steps", mode="before")
    @classmethod
    def coerce(cls, v: Any) -> list[dict]:
        # 旧形式 list[str] / 各種キー名 / 依存付き dict を一様な {description, instruction,
        # depends_on} へ。depends_on / deps / after / requires / dependencies のいずれのキーでも依存を拾う。
        out: list[dict] = []
        for item in v if isinstance(v, list) else []:
            if isinstance(item, str) and item.strip():
                out.append(
                    {"id": None, "description": item.strip(), "instruction": "", "depends_on": []}
                )
            elif isinstance(item, dict):
                desc = None
                for k in ("description", "step", "task", "content"):
                    if isinstance(item.get(k), str) and item[k].strip():
                        desc = item[k].strip()
                        break
                if desc is None:
                    continue
                instruction = item["instruction"].strip() if isinstance(item.get("instruction"), str) else ""
                deps: list = []
                for dk in ("depends_on", "deps", "after", "requires", "dependencies"):
                    if dk in item:
                        deps = _coerce_int_list(item.get(dk))
                        break
                out.append(
                    {
                        "id": item.get("id"),
                        "description": desc,
                        "instruction": instruction,
                        "depends_on": deps,
                    }
                )
        return out


class VerdictSchema(BaseModel):
    verdict: Literal["pass", "retry", "replan"]
    feedback: str = ""

    @field_validator("feedback", mode="before")
    @classmethod
    def coerce_feedback(cls, v: Any) -> str:
        return v if isinstance(v, str) else ""


class RouteSchema(BaseModel):
    """orchestrator のルーティング判定 (構造化出力用)。"""

    route: Literal["direct", "plan"] = Field(
        description="direct=1回の回答や1〜2ツールで完結する単純な要求 / plan=多段・複数ツールの複雑な要求"
    )


class EntrySelectionSchema(BaseModel):
    """PlanStep.data の 1 エントリ (= 1 ツール出力) に対する「残す箇所」の選択指定。

    LLM は値を生成せず、どのフィールド/項目を残すかだけを指定する。実抽出はコード側
    (common.apply_data_selection) が元データから射影するため、値は元のまま保たれる。
    """

    index: int  # data リスト内のエントリ位置 (0 始まり)
    keep_fields: list[str] = []  # 各 dict で残すキー名 (空 = 全フィールド)
    keep_items: list[int] | None = None  # artifact が list のとき残す項目位置 (None = 全項目)

    @field_validator("index", mode="before")
    @classmethod
    def coerce_index(cls, v: Any) -> int | None:
        # None を返すと int 必須違反で ValidationError → DataSelectionSchema 側で当該エントリを捨てる
        return _coerce_opt_int(v)

    @field_validator("keep_fields", mode="before")
    @classmethod
    def coerce_keep_fields(cls, v: Any) -> list[str]:
        return [s for s in v if isinstance(s, str) and s] if isinstance(v, list) else []

    @field_validator("keep_items", mode="before")
    @classmethod
    def coerce_keep_items(cls, v: Any) -> list[int] | None:
        return None if v is None else _coerce_int_list(v)


class DataSelectionSchema(BaseModel):
    """スクリーニングの選択結果 (エントリごとの選択指定の集合)。"""

    selections: list[EntrySelectionSchema] = []

    @field_validator("selections", mode="before")
    @classmethod
    def drop_malformed(cls, v: Any) -> list[dict]:
        # dict 以外や index を欠く要素を捨て、堅牢に拾えるものだけ通す。
        out: list[dict] = []
        for item in v if isinstance(v, list) else []:
            if isinstance(item, dict) and _coerce_opt_int(item.get("index")) is not None:
                out.append(item)
        return out


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


async def structured_or_parse(
    model: Any,
    messages: list[BaseMessage],
    schema: type[T],
    *,
    use_structured: bool,
    fallback: T | Callable[[], T],
    max_retries: int = 2,
) -> T:
    """構造化出力が使えるなら with_structured_output、不可ならテキストパースで schema を得る。

    - use_structured=True (openai 等): with_structured_output でスキーマ準拠を保証。
      失敗 (例外・None) してもテキストパースへフォールバックし、最終的に fallback を返す。
    - use_structured=False (ollama 等): 従来どおり parse_with_retry。
    全経路で例外を投げず、必ず schema インスタンス (最悪 fallback) を返す契約。
    """
    if use_structured:
        try:
            result = await model.with_structured_output(schema).ainvoke(messages)
            if isinstance(result, schema):
                return result
            logger.warning(
                "構造化出力が想定外の型を返したためテキストパースへフォールバックします (schema=%s)",
                schema.__name__,
            )
        except Exception:
            logger.exception(
                "構造化出力に失敗したためテキストパースへフォールバックします (schema=%s)",
                schema.__name__,
            )
    return await parse_with_retry(
        model, messages, schema, max_retries=max_retries, fallback=fallback
    )
