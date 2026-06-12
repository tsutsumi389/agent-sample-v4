# 汎用AIエージェント — アーキテクチャ設計書

## 1. 概要とゴール

ローカル LLM (Ollama / gpt-oss) を頭脳とする、チャットベースの汎用AIエージェントを構築する。LangGraph 1.x の `create_agent` をエージェントランタイムに採用し、会話スレッド (短期記憶) を PostgreSQL チェックポインタに、ユーザーごとの長期記憶 (LangMem) を pgvector ベクトルインデックス付き PostgreSQL Store に永続化する。使えば使うほどユーザーに最適化される (バックグラウンドのメモリ統合)。

**設計の核となる要件:**
- **拡張性 (最重要):** ネイティブツールは `backend/app/tools/` にファイルを置くだけで自動登録。MCP サーバは `mcp_servers.json` に書くだけで追加。**いずれもコア (`graph.py` 等) を一切変更しない。**
- **ストリーミング:** SSE でトークン・ツール呼び出し・ツール結果・進捗をリアルタイム配信。
- **user_id の貫通:** v1 はシングルユーザーだが、API・config・namespace すべてに `user_id` を通す。
- **シンプルさ:** 認証なし。テストは数本のサニティテストのみ。過度な抽象化はしない。ただしツールレジストリと MCP ローダーは本物のプラガブル設計にする。

## 2. 技術スタックの決定 (すべて確定)

| 領域 | 採用 | 理由 |
|------|------|------|
| パッケージ管理 | **uv** | 要件指定。`uv run` / `uv sync` で運用 |
| Python | **3.12** | 全依存が `>=3.10`、3.12 で `get_stream_writer` の contextvars 問題も無し |
| エージェント | **`langchain.agents.create_agent`** (langchain 1.3.8) | `create_react_agent` は非推奨。middleware / ToolRuntime / store / checkpointer を直接受け取る |
| Web | **FastAPI 0.136.3 + uvicorn[standard] 0.49.0** | FastAPI 0.135+ ネイティブSSE (`fastapi.sse.EventSourceResponse`) を採用、sse-starlette 不要 |
| LLM (チャット+ツール) | **ChatOllama `gpt-oss`** | `num_ctx=32768`, `reasoning="medium"` |
| LLM (メモリ抽出) | **ChatOllama `qwen3`** | gpt-oss は構造化出力 (trustcall) が不安定。抽出は tool-calling が堅牢な qwen3 を採用 (`reasoning="low"`) |
| 埋め込み | **OllamaEmbeddings `nomic-embed-text`** | **768次元** (pgvector / store index の dims=768 に厳密一致) |
| DB | **PostgreSQL (pgvector/pgvector:pg17)** | Store のベクトルインデックスに pgvector が必須。プレーン postgres では `CREATE EXTENSION vector` が失敗する |
| チェックポインタ | **AsyncPostgresSaver** (langgraph-checkpoint-postgres 3.1.0) | 会話スレッドの永続化 |
| 長期記憶 Store | **AsyncPostgresStore** (同パッケージ) | index={dims:768, embed:nomic} でセマンティック検索 |
| 長期記憶ロジック | **LangMem 0.0.30** | hot-path ツール (manage/search) + バックグラウンド統合 (ReflectionExecutor) |
| MCP | **langchain-mcp-adapters 0.3.0** | `MultiServerMCPClient` で MCP→BaseTool 変換 |
| フロント | **React 19.2.7 + TypeScript 5.9 + Vite 8** | チャットUIは外部UIライブラリ不要。`fetch + ReadableStream` で SSE 受信 |

> **TypeScript は 5.9 を採用** (6.0 でも可だが、エコシステム互換性を最優先し安全側に倒す。7.0 beta は不採用)。

## 3. 全体アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│  Frontend (Vite dev :5173 → /api を :8000 へ proxy)          │
│  ┌───────────┐ ┌──────────────┐ ┌─────────────────┐          │
│  │ Sidebar   │ │ Chat (SSE)   │ │ Memory Panel    │          │
│  │ threads   │ │ stream/tools │ │ what AI remembers│         │
│  └───────────┘ └──────────────┘ └─────────────────┘          │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP / SSE (POST + fetch ReadableStream)
┌────────────────────────▼────────────────────────────────────┐
│  Backend  FastAPI (uvicorn)                                  │
│  routers/  chat(SSE) · threads · memory · health · tools     │
│  agent/    graph.py (graph factory) · prompts.py             │
│  services/ streaming.py · history.py · threads.py            │
│  memory/   manager.py · tools.py (LangMem)                   │
│  mcp/      loader.py (MCP loader)                            │
│  tools/    registry.py + 自動探索される @tool 群             │
│            (web_search.py, datetime.py, calculator.py ...)   │
│  core/     config.py (pydantic-settings) · db.py (pool)      │
│                                                              │
│  app.state.agent  = create_agent(...)  ← 起動時にビルド      │
│  app.state.pool   = AsyncConnectionPool (lifespan所有)       │
│  app.state.reflection_executor (LangMem 背景統合)            │
└──────┬──────────────────────────────────┬───────────────────┘
       │ psycopg AsyncConnectionPool       │ HTTP
┌──────▼──────────────────┐        ┌───────▼──────────────────┐
│ PostgreSQL (pgvector)   │        │ Ollama (:11434)          │
│ - checkpoints (threads) │        │ gpt-oss / qwen3          │
│ - store + vector(768)   │        │ nomic-embed-text         │
│   namespace per user_id │        └──────────────────────────┘
└─────────────────────────┘
```

## 4. コンポーネント設計

### 4.1 DB接続 (core/db.py)
FastAPI lifespan で **単一の `AsyncConnectionPool`** を所有し、checkpointer / store / agent で共有する。`from_conn_string` は接続を context 終了時に閉じてしまうため**使わない**。接続 kwargs は必須:
```python
kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row}
```
起動時に `await checkpointer.setup()` と `await store.setup()` を一度だけ呼ぶ (冪等・マイグレーション対応。テーブルは自動生成されない)。

### 4.2 ツールレジストリ (tools/registry.py) — プラガブルの中核
`backend/app/tools/` パッケージを `pkgutil.iter_modules` でスキャンし、各モジュール内の module-level `BaseTool` インスタンス (= `@tool` デコレート済みオブジェクト) をすべて登録する。**ファイルを1つ置くだけでツールが増える。**コア変更不要。`registry.py` / `base.py` / `__init__.py` は除外。さらに `importlib.metadata.entry_points(group="agent.tools")` でサードパーティ製プラグインパッケージ (uv add でインストール) も探索する。ツール名の重複は登録時に検出してエラーにする。

### 4.3 MCP ローダー (mcp/loader.py) — もう一つのプラガブルの中核
`mcp_servers.json` (Claude Desktop 互換の `{"mcpServers": {...}}` 形状) を読み、`${ENV_VAR}` を環境変数で置換し、transport を自動補完 (`command` あれば `stdio`、`url` あれば `http`) して `MultiServerMCPClient` に渡す。`tool_name_prefix=True` でサーバ名プレフィックスを付け、ネイティブツールとの名前衝突を防ぐ。`get_tools()` はステートレス (呼び出しごとに新規セッション) なので起動時に一括取得して問題ない。**JSONに1エントリ足すだけで MCP サーバが追加される。**コア変更不要。

### 4.4 エージェントファクトリ (agent/graph.py)
```
all_tools = registry.all()                       # ネイティブ (自動探索)
          + langmem_hotpath_tools()              # manage/search memory
          + await mcp_client.get_tools()         # MCP (JSON設定)
agent = create_agent(
    model=ChatOllama(gpt-oss, num_ctx=32768, reasoning="medium", temperature=0),
    tools=all_tools,
    system_prompt=SYSTEM_PROMPT,                 # 起動時にユーザープロフィールを注入する版は middleware で
    context_schema=AgentContext(user_id),
    checkpointer=checkpointer, store=store,
)
```
- `AgentContext` は `@dataclass` で `user_id` を持つ。invoke 時に `context=AgentContext(user_id=...)` で渡す (config の `configurable` ではなく `context`)。
- ただし LangMem の namespace テンプレート `{langgraph_user_id}` は **config から** 解決されるため、invoke 時には config に `configurable.langgraph_user_id` も併せて渡す (両方プラミングする)。

### 4.5 長期記憶 (memory/manager.py + memory/tools.py) — LangMem
**ハイブリッド (collection-style) 採用:**
- **hot-path ツール:** `create_manage_memory_tool` / `create_search_memory_tool`、namespace=`("memories", "{langgraph_user_id}")`。エージェントが会話中に明示的に記憶を保存/検索できる (= メモリパネルの可視性とツール呼び出し可視性に直結)。
- **バックグラウンド統合:** `create_memory_store_manager(ChatOllama("qwen3"), namespace=("memories","{langgraph_user_id}"), enable_inserts=True)` を `ReflectionExecutor` でラップ。各チャットターン完了後に `executor.submit({"messages": [...]}, after_seconds=30, config={"configurable":{"langgraph_user_id":user_id}})` を呼ぶ (debounce: 新しい submit が保留中タスクをキャンセル)。
- メモリパネル API は `store.asearch(("memories", user_id), ...)` で記憶一覧を返す。

> **gpt-oss を抽出に使わない**のは、trustcall (構造化出力) が gpt-oss + Ollama で不安定 (langchain#33116) なため。抽出専用に qwen3 を立てる。スキーマはフラットに保つ。

> **背景統合の限界 (明記):** ReflectionExecutor はインプロセスのバックグラウンドスレッドで動くため、プロセスが落ちると保留中の debounce タスクは失われる (best-effort)。v1 では許容する。

### 4.6 ストリーミングブリッジ (services/streaming.py)
`agent.astream(..., stream_mode=["messages","updates","custom"], version="v2")` を使い、統一された `StreamPart` dict (`{"type","ns","data"}`) を SSE イベントへ変換する。`messages` → token、`updates` → tool_call / tool_result、`custom` → progress。`version="v2"` に統一し、v1 のタプル形式とは混在させない。

## 5. データモデル / 永続化

| 種別 | 場所 | キー | 内容 |
|------|------|------|------|
| 会話スレッド | checkpoints テーブル群 (saver.setup()) | `thread_id` | LangGraph の state / messages 履歴 |
| 長期記憶 | store テーブル + vector(768) | namespace=`("memories", user_id)`, key=uuid | `{"kind":..., "content": {...}}` (LangMem 管理) |

スレッドのメタ情報 (タイトル・作成日時) は専用テーブルを作らず、**最小限の `threads` テーブル**を1つだけ追加して管理する (id, user_id, title, created_at, updated_at)。チェックポインタには無いため。タイトルは最初のユーザーメッセージ先頭40文字から自動生成。

## 6. エラーハンドリング方針
- SSE 中の例外は `event: error` で client に通知してからストリームを閉じる。
- MCP の `handle_tool_errors=True` (デフォルト) によりツールエラーは `ToolMessage(status="error")` となりエージェントが自己修復可能。
- Ollama モデル未 pull は `validate_model_on_init=True` で起動時に fail-fast。
- `num_ctx` は必ず明示 (デフォルト2048ではエージェントが沈黙切り捨てされる)。

## 7. セットアップ時の注意 (README / Makefile に反映)
- **gpt-oss の再 pull が必須**: 2026年2月以前に pull したモデルはツール (Optional/Union 引数) でHTTP500になるチャットテンプレートバグがある。`ollama pull gpt-oss` で更新。
- Ollama サーバは **>=0.11** (streaming+tools, gpt-oss 要件)。
- pgvector イメージ必須 (`pgvector/pgvector:pg17`)。

## 8. 非対象 (v1スコープ外)
認証 / 認可、マルチワーカーでの SSE ブロードキャスト (Redis pub/sub)、ツールのランタイムホットリロード (再起動で反映する方針)、本番リバースプロキシ調整、重厚なテストスイート。
