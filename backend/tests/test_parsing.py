"""parsing.py (堅牢 JSON パース) のテスト。"""

from langchain_core.messages import HumanMessage

from app.agent.parsing import (
    PlanSchema,
    VerdictSchema,
    extract_json,
    parse_json_as,
    parse_with_retry,
    strip_think,
)
from tests.fakes import ScriptedModel


def test_strip_think():
    assert strip_think("<think>内部思考</think>PLAN") == "PLAN"
    assert strip_think("そのまま") == "そのまま"


def test_extract_json_plain():
    assert extract_json('{"steps": ["a"]}') == {"steps": ["a"]}


def test_extract_json_with_think_and_noise():
    text = '<think>考え中...</think>結果は以下です: {"verdict": "pass", "feedback": ""} 以上'
    assert extract_json(text) == {"verdict": "pass", "feedback": ""}


def test_extract_json_code_fence():
    text = '説明\n```json\n{"steps": ["x", "y"]}\n```\n以上'
    assert extract_json(text) == {"steps": ["x", "y"]}


def test_extract_json_nested_braces():
    text = 'プレフィックス {"a": {"b": 1}} サフィックス'
    assert extract_json(text) == {"a": {"b": 1}}


def test_extract_json_broken_inputs():
    assert extract_json('{"a": ') is None  # 途中切断
    assert extract_json("") is None
    assert extract_json("JSONなし") is None
    assert extract_json(None) is None  # type: ignore[arg-type]


def test_plan_schema_coerces_dict_steps():
    parsed = parse_json_as(
        '{"steps": [{"description": "調査する"}, "まとめる", {"task": "報告する"}, "", 42]}',
        PlanSchema,
    )
    assert parsed is not None
    assert [s.description for s in parsed.steps] == ["調査する", "まとめる", "報告する"]


def test_plan_schema_extracts_dependencies():
    parsed = parse_json_as(
        '{"steps": ['
        '{"id": 1, "description": "A", "depends_on": []}, '
        '{"description": "B", "deps": ["1"]}, '  # 別キー名 + 数字文字列も拾う
        '{"description": "C", "depends_on": [1, 2]}'
        "]}",
        PlanSchema,
    )
    assert parsed is not None
    assert [s.depends_on for s in parsed.steps] == [[], [1], [1, 2]]


def test_verdict_schema_rejects_unknown_verdict():
    assert parse_json_as('{"verdict": "banana"}', VerdictSchema) is None


def test_verdict_schema_coerces_missing_feedback():
    parsed = parse_json_as('{"verdict": "retry", "feedback": null}', VerdictSchema)
    assert parsed is not None
    assert parsed.verdict == "retry"
    assert parsed.feedback == ""


async def test_parse_with_retry_recovers_on_second_attempt():
    model = ScriptedModel(["これはJSONではない", '{"verdict": "pass", "feedback": ""}'])
    result = await parse_with_retry(
        model,
        [HumanMessage(content="判定して")],
        VerdictSchema,
        fallback=VerdictSchema(verdict="retry", feedback="fb"),
    )
    assert result.verdict == "pass"
    assert len(model.calls) == 2
    # リトライ時に修正指示が追記されている
    assert any("JSONとして不正" in m.content for m in model.calls[1])


async def test_parse_with_retry_returns_fallback_when_all_fail():
    model = ScriptedModel(["壊れた出力", "また壊れた出力"])
    result = await parse_with_retry(
        model,
        [HumanMessage(content="判定して")],
        VerdictSchema,
        fallback=VerdictSchema(verdict="pass", feedback=""),
    )
    assert result.verdict == "pass"


async def test_parse_with_retry_survives_model_exceptions():
    model = ScriptedModel([RuntimeError("接続断"), RuntimeError("接続断")])
    result = await parse_with_retry(
        model,
        [HumanMessage(content="判定して")],
        PlanSchema,
        fallback=lambda: PlanSchema(steps=["fallback"]),
    )
    assert [s.description for s in result.steps] == ["fallback"]
