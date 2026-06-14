# 汎用AIエージェント (LangGraph + Ollama + LangMem)

ローカル LLM (Ollama / gpt-oss) を頭脳とする、チャットベースの汎用AIエージェントです。

- **エージェントランタイム:** LangGraph 1.x (`langchain.agents.create_agent`)
- **短期記憶:** 会話スレッドを PostgreSQL チェックポインタ (AsyncPostgresSaver) に永続化
- **長期記憶:** LangMem + pgvector 付き PostgreSQL Store。使えば使うほどユーザーに最適化 (バックグラウンドのメモリ統合)
- **ストリーミング:** SSE でトークン・ツール呼び出し・ツール結果・進捗をリアルタイム配信
- **拡張性 (最重要):** ツール追加・MCP サーバ追加でコアのコード変更が一切不要 → [拡張方法](#拡張方法-最重要)

## アーキテクチャ

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
│  core/     config.py (pydantic-settings) · db.py (pool)      │
└──────┬──────────────────────────────────┬───────────────────┘
       │ psycopg AsyncConnectionPool       │ HTTP
┌──────▼──────────────────┐        ┌───────▼──────────────────┐
│ PostgreSQL (pgvector)   │        │ Ollama (:11434)          │
│ - checkpoints (threads) │        │ gpt-oss / qwen3          │
│ - store + vector(768)   │        │ nomic-embed-text         │
└─────────────────────────┘        └──────────────────────────┘
```

詳細は **設計書サイト** [docs/index.html](docs/index.html) を参照してください（ブラウザで開くと、図解つきの設計書を読めます）。主要ページ:

- [はじめに（全体像と用語）](docs/guide/01-overview.html)
- [システムアーキテクチャ](docs/guide/02-architecture.html)
- [マルチエージェント設計](docs/guide/03-multi-agent.html)
- [記憶アーキテクチャ](docs/guide/04-memory.html)
- [ストリーミング設計](docs/guide/05-streaming.html)
- [拡張ガイド（ツール / MCP）](docs/guide/06-extensibility.html)
- [API リファレンス](docs/guide/07-api-reference.html)
- [データモデルと永続化](docs/guide/08-data-model.html)
- [フロントエンド設計](docs/guide/09-frontend.html)
- [セットアップと運用](docs/guide/10-setup.html)

## 前提

| ツール | バージョン |
|--------|-----------|
| Docker (compose v2) | 最新推奨 |
| uv | 最新推奨 |
| Node.js | **20.19+ または 22.12+** (Vite 8 要件) |
| Ollama | **>= 0.11** (gpt-oss の streaming + tools 要件) |

## クイックスタート

```bash
# 1. 環境変数ファイルを用意
cp .env.example .env

# 2. 初回セットアップ (Postgres 起動 + Ollama モデル pull + 依存同期)
make setup

# 3. 起動 (Postgres → バックエンド + フロントエンドを並列起動)
make dev
```

ブラウザで http://localhost:5173 を開いてください。

別々のターミナルで起動したい場合は、`make dev` の代わりに:

```bash
make backend    # ターミナル1: FastAPI (:8000)
make frontend   # ターミナル2: Vite dev server (:5173)
```

> **重要: gpt-oss の再 pull が必要です。**
> 2026年2月以前に pull した gpt-oss モデルには、ツール呼び出し (Optional/Union 引数) で
> HTTP 500 になるチャットテンプレートのバグがあります。古いモデルをお持ちの場合は
> `ollama pull gpt-oss` で更新してください (`make models` に含まれています)。

## 拡張方法 (最重要)

コア (`graph.py` 等) を一切変更せずにエージェントの能力を拡張できます。

### 1. ネイティブツールの追加

`backend/app/tools/` にファイルを1つ置くだけです。module-level の `@tool` デコレート済み
オブジェクトが起動時に自動探索・自動登録されます。

```python
# backend/app/tools/your_tool.py
from langchain_core.tools import tool

@tool
def your_tool(query: str) -> str:
    """ツールの説明 (LLM がこれを読んで使い方を判断します)。"""
    return f"result for {query}"
```

バックエンドを再起動すると反映されます。レジストリへの手動登録は不要です。

### 2. MCP サーバの追加

`backend/mcp_servers.json` (Claude Desktop 互換形式) に1エントリ追加するだけです。
例: [time サーバ](https://pypi.org/project/mcp-server-time/) を追加する場合:

```json
{
  "mcpServers": {
    "time": {
      "command": "uvx",
      "args": ["mcp-server-time"]
    }
  }
}
```

- **stdio サーバ:** `command` / `args` を指定 (transport は自動補完)
- **HTTP サーバ:** `url` を指定
- **シークレット:** 値の中で `${ENV_VAR}` と書くと環境変数 (`.env`) で置換されます

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}" }
    }
  }
}
```

バックエンドを再起動すると反映されます。MCP ツールには `サーバ名__` プレフィックスが付き、
ネイティブツールとの名前衝突を防ぎます。

## API 一覧

| メソッド | パス | 説明 |
|---------|------|------|
| GET | `/api/health` | ヘルスチェック |
| GET | `/api/threads?user_id=` | スレッド一覧 (更新日時降順) |
| POST | `/api/threads` | スレッド作成 (**201** を返す) |
| GET | `/api/threads/{thread_id}/messages?user_id=` | 会話履歴 |
| DELETE | `/api/threads/{thread_id}?user_id=` | スレッド削除 |
| POST | `/api/chat/stream` | チャット送信 (**SSE**: token / tool_call / tool_result / progress / done / error) |
| GET | `/api/memory?user_id=&query=&limit=` | 長期記憶の一覧 / セマンティック検索 |
| DELETE | `/api/memory/{key}?user_id=` | 記憶の削除 |
| GET | `/api/tools` | 登録済みツール一覧 (native / langmem / mcp) |

- `POST /api/chat/stream` の `thread_id` は **事前に `POST /api/threads` で作成済み**である必要があります。未知の `thread_id` は SSE 開始前に HTTP 404 `{"detail": "thread not found"}` を返します。
- フィールド名・イベント形式の詳細は [API リファレンス](docs/guide/07-api-reference.html) を参照してください。

## 設計上の決定事項

- **gpt-oss = チャット + ツール呼び出し** (`num_ctx=32768`, `reasoning="medium"`)
- **qwen3 = メモリ抽出専用** — gpt-oss は構造化出力 (trustcall) が不安定なため、抽出はツール呼び出しが堅牢な qwen3 を使用
- **nomic-embed-text = 埋め込み (768次元)** — pgvector / store index の `dims=768` に厳密一致
- **pgvector/pgvector:pg17 イメージ必須** — プレーン postgres では `CREATE EXTENSION vector` が失敗
- **user_id の貫通** — v1 はシングルユーザー (`"default-user"` 固定) だが、API・config・namespace すべてに `user_id` を通す
- **SSE は fetch + ReadableStream で受信** — POST body が必要なため native EventSource は不使用

## 既知の制約

- **バックグラウンドのメモリ統合は best-effort** — ReflectionExecutor はインプロセスで動くため、プロセスが落ちると保留中の debounce タスクは失われます
- **シングルユーザー v1** — `user_id` は配線済みですが、UI は単一ユーザー前提です
- **認証なし** — ローカル開発用途を想定しています

## Makefile ターゲット

```
make help       ターゲット一覧
make up         Postgres 起動 (healthcheck 待機)
make down       Postgres 停止
make models     Ollama モデル pull
make setup      初回セットアップ
make dev        Postgres + backend + frontend 起動
make backend    バックエンドのみ起動
make frontend   フロントエンドのみ起動
make test       バックエンドのテスト
make db-shell   psql 接続
make logs       Postgres ログ
make clean      Postgres 停止 + データ削除
```
