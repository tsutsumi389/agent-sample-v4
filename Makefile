.PHONY: help up down logs db-shell models backend frontend dev setup test clean

help: ## ターゲット一覧を表示
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

up: ## Postgres 起動 (healthcheck 通過まで待機)
	docker compose up -d --wait

down: ## Postgres 停止
	docker compose down

logs: ## Postgres ログ表示
	docker compose logs -f postgres

db-shell: ## psql で DB に接続
	docker compose exec postgres psql -U postgres -d agent

models: ## 必要な Ollama モデルを pull (gpt-oss は再pullでテンプレ修正)
	ollama pull gpt-oss
	ollama pull qwen3
	ollama pull nomic-embed-text

backend: ## バックエンド起動 (依存同期込み)
	cd backend && uv sync && uv run uvicorn app.main:app --reload --port 8000

frontend: ## フロントエンド起動 (依存インストール込み)
	cd frontend && npm install && npm run dev

dev: up ## Postgres 起動後、バックエンド+フロントエンドを並列起動
	$(MAKE) -j2 backend frontend

setup: up models ## 初回セットアップ (Postgres + モデル pull + 依存同期)
	cd backend && uv sync
	cd frontend && npm install

test: ## バックエンドのテスト実行
	cd backend && uv run pytest -q

clean: ## Postgres 停止 + データ削除
	docker compose down -v
