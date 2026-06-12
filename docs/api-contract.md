## API契約 (バックエンド・フロント独立実装の唯一の真実)

ベースURL: 開発時はフロントから相対パス `/api/...` (Vite proxy → `http://localhost:8000`)。全レスポンスは `Content-Type: application/json` (SSE を除く)。`user_id` は v1 では固定値 `"default-user"` をフロントが送る (将来差し替え可能なよう全エンドポイントで受ける)。

### 共通エラー形状 (非SSE)
HTTP 4xx/5xx 時:
```json
{ "detail": "human readable message" }
```

---

### 1. GET /api/health
ヘルスチェック。
- レスポンス 200:
```json
{ "status": "ok" }
```

---

### 2. GET /api/threads?user_id={user_id}
スレッド一覧 (更新日時の降順)。
- レスポンス 200:
```json
{
  "threads": [
    {
      "thread_id": "t_3f9a...",
      "title": "東京の天気について",
      "created_at": "2026-06-12T09:00:00Z",
      "updated_at": "2026-06-12T09:05:00Z"
    }
  ]
}
```

### 3. POST /api/threads
新規スレッド作成。**HTTP 201** を返す (他のエンドポイントはすべて 200)。
- リクエスト body:
```json
{ "user_id": "default-user", "title": null }
```
(`title` は省略/null 可。null の場合サーバは `"新しい会話"` を初期タイトルにする)
- レスポンス 201:
```json
{
  "thread_id": "t_3f9a...",
  "title": "新しい会話",
  "created_at": "2026-06-12T09:00:00Z",
  "updated_at": "2026-06-12T09:00:00Z"
}
```

### 4. GET /api/threads/{thread_id}/messages?user_id={user_id}
スレッドの会話履歴 (チェックポインタから復元)。
- レスポンス 200:
```json
{
  "thread_id": "t_3f9a...",
  "messages": [
    { "role": "user", "content": "東京の天気は？", "id": "m1" },
    {
      "role": "assistant",
      "content": "東京は晴れです。",
      "id": "m2",
      "tool_calls": [
        { "id": "call_abc", "name": "web_search", "args": { "query": "東京 天気" } }
      ]
    },
    {
      "role": "tool",
      "content": "晴れ, 25℃",
      "id": "m3",
      "tool_call_id": "call_abc",
      "name": "web_search"
    }
  ]
}
```
- 履歴が無い (未送信) スレッドは `"messages": []`。

### 5. DELETE /api/threads/{thread_id}?user_id={user_id}
スレッド削除 (threads テーブルから削除。チェックポイントは残置で可)。
- レスポンス 200: `{ "deleted": true }`

---

### 6. POST /api/chat/stream  ← SSE
チャット送信。**Server-Sent Events** を返す (`Content-Type: text/event-stream`)。
フロントは **native EventSource ではなく** `fetch` + `ReadableStream` で受信する (POST body が必要なため)。

- リクエスト headers: `Content-Type: application/json`, `Accept: text/event-stream`
- リクエスト body:
```json
{ "message": "東京の天気は？", "thread_id": "t_3f9a...", "user_id": "default-user" }
```

**`thread_id` は POST /api/threads で作成済みのスレッドを参照しなければならない。**
未知の `thread_id` の場合、SSE バイトを一切送る前に HTTP 404 `{"detail": "thread not found"}` を返す。
スレッドのタイトルは最初のユーザーメッセージ (先頭40文字) から **ストリーミング開始前に** 設定される。

- レスポンス: SSE ストリーム。**名前付きイベント**。各イベントの `data` はフラットなJSON文字列1行。
  サーバは15秒ごとに `:` から始まる keep-alive コメント行を送る (フロントは無視する)。

**イベント定義 (フィールド名は厳密):**

| event | data (JSON) | 発生タイミング |
|-------|-------------|----------------|
| `token` | `{"content": "Hel", "node": "model"}` | LLMトークン差分ごと (`content` は差分文字列、`node` は emit ノード名) |
| `tool_call` | `{"id": "call_abc", "name": "web_search", "args": {"query":"..."}}` | アシスタントがツール呼び出しを確定したとき |
| `tool_result` | `{"id": "call_abc", "name": "web_search", "content": "晴れ, 25℃"}` | ツール実行結果が出たとき (`content` は最大2000字に切り詰め) |
| `progress` | `{"status": "memories を検索中..."}` | (任意) custom writer からの進捗 |
| `done` | `{"thread_id": "t_3f9a...", "title": "東京の天気について"}` | ストリーム正常終了。`title` はサーバが更新した最新タイトル (初回は自動生成、無変更なら既存値) |
| `error` | `{"message": "..."}` | 例外発生。直後にストリームを閉じる |

**SSEワイヤ形式の例 (実バイト):**
```
event: tool_call
data: {"id":"call_abc","name":"web_search","args":{"query":"東京 天気"}}

event: token
data: {"content":"東京","node":"model"}

event: done
data: {"thread_id":"t_3f9a...","title":"東京の天気について"}

```
(各イベントは空行で区切られる。`data:` の値は単一行JSON。)

**フロントのパース規約:** UTF-8マルチバイトは `TextDecoder({stream:true})` で結合。`event:` 行でイベント名、`data:` 行で本文を蓄積、空行で1イベント確定・ディスパッチ。`:` 始まりの行 (keep-alive) は無視。

---

### 7. GET /api/memory?user_id={user_id}&query={optional}&limit={20}
エージェントがそのユーザーについて記憶している内容 (メモリパネル用)。
- `query` 省略時は最新の記憶を新しい順に返す。指定時はセマンティック検索。
- レスポンス 200:
```json
{
  "user_id": "default-user",
  "memories": [
    {
      "key": "mem_8a2c...",
      "content": "ユーザーはダークモードを好み、日本語で回答してほしい",
      "namespace": ["memories", "default-user"],
      "updated_at": "2026-06-12T09:05:00Z",
      "score": 0.82
    }
  ]
}
```
(`content` は store 値から抽出した人間可読テキスト。LangMem 形式 `{"kind":...,"content":{...}}` の場合は `content` フィールドを文字列化。`score` は検索時のみ、一覧時は null。)

### 8. DELETE /api/memory/{key}?user_id={user_id}
特定の記憶を削除 (メモリパネルから手動削除)。
- レスポンス 200: `{ "deleted": true }`

---

### 9. GET /api/tools
登録済みツール一覧 (ネイティブ + MCP。デバッグ/可視化用)。
- レスポンス 200:
```json
{
  "tools": [
    { "name": "web_search", "description": "Search the web.", "source": "native" },
    { "name": "manage_memory", "description": "...", "source": "langmem" },
    { "name": "filesystem_read_file", "description": "...", "source": "mcp" }
  ]
}
```

---

### CORS / 開発
- バックエンドは `CORSMiddleware` で `allow_origins=["http://localhost:5173"]` を許可 (Vite proxy を使う場合でも保険として設定)。
- フロントは相対パス `/api/...` を叩き、`vite.config.ts` の `server.proxy` で `:8000` へ転送 (CORS回避)。
