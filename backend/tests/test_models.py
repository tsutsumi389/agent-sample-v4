"""default_model_factory のプロバイダ分岐テスト (ネットワーク不要)。

openai 分岐は ChatOpenAI を構築するだけで API 呼び出しは行わないため、API キーは
ダミーで足りる。ollama 分岐は validate_model_on_init=True でサーバ接続するため、ここでは
検証せず (構築自体がネットワークを要する)、型選択のみを openai 経路で確認する。
"""

from langchain_openai import ChatOpenAI

from app.agent.models import default_model_factory
from app.core.config import Settings

OPENAI = dict(
    llm_provider="openai",
    openai_api_key="x",
    openai_control_model="ctl-model",
    openai_chat_model="cht-model",
    openai_base_url="http://compat.local/v1",
)


def test_factory_openai_control_selects_control_model():
    model = default_model_factory("control", Settings(**OPENAI), ["nostream"])
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "ctl-model"


def test_factory_openai_chat_selects_chat_model():
    model = default_model_factory("chat", Settings(**OPENAI), [])
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "cht-model"


def test_factory_openai_propagates_base_url():
    model = default_model_factory("chat", Settings(**OPENAI), [])
    # 互換エンドポイントが ChatOpenAI に伝搬している
    assert str(model.openai_api_base).startswith("http://compat.local")


def test_factory_default_provider_is_ollama():
    # 既定は ollama。supports_structured_output が False であることで分岐を確認
    # (ChatOllama 構築はネットワークを要するため型生成はしない)。
    assert Settings().llm_provider == "ollama"
    assert Settings().supports_structured_output is False
    assert Settings(**OPENAI).supports_structured_output is True
