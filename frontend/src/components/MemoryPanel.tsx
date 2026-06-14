import { useCallback, useEffect, useRef, useState } from "react";
import {
  deleteMemory,
  forgetConfirm,
  forgetPreview,
  listMemories,
} from "../api/client";
import type { ForgetCandidate, Memory } from "../api/types";

interface Props {
  userId: string;
  /** インクリメントされるたびに再取得する (チャットの done イベント後など)。 */
  refreshKey: number;
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleString("ja-JP", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** スコープ付き一括忘却の確認モーダルの状態。 */
interface ForgetState {
  query: string;
  candidates: ForgetCandidate[];
  selected: Set<string>;
}

export function MemoryPanel({ userId, refreshKey }: Props) {
  const [memories, setMemories] = useState<Memory[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [forgetState, setForgetState] = useState<ForgetState | null>(null);
  const [forgetting, setForgetting] = useState(false);
  const [previewing, setPreviewing] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const activeQueryRef = useRef("");

  const load = useCallback(
    async (q: string) => {
      setLoading(true);
      setError(null);
      setNotice(null);
      try {
        const res = await listMemories(userId, q.trim() || undefined);
        setMemories(res.memories);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    },
    [userId],
  );

  useEffect(() => {
    void load(activeQueryRef.current);
  }, [load, refreshKey]);

  const handleSearch = () => {
    activeQueryRef.current = query;
    void load(query);
  };

  const handleDelete = async (key: string) => {
    try {
      await deleteMemory(key, userId);
      await load(activeQueryRef.current);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  // 「まとめて忘れる」: 候補を取得して確認モーダルを開く (この時点では削除しない)
  const handleForgetPreview = async () => {
    const q = query.trim();
    if (!q || previewing) return; // 連打による多重リクエストを防ぐ
    setPreviewing(true);
    setError(null);
    setNotice(null);
    try {
      const res = await forgetPreview(userId, q);
      if (res.candidates.length === 0) {
        setNotice(`「${q}」に関連する記憶は見つかりませんでした。`);
        return;
      }
      setForgetState({
        query: q,
        candidates: res.candidates,
        selected: new Set(res.candidates.map((c) => c.key)),
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPreviewing(false);
    }
  };

  const toggleSelected = (key: string) => {
    setForgetState((prev) => {
      if (!prev) return prev;
      const selected = new Set(prev.selected);
      if (selected.has(key)) selected.delete(key);
      else selected.add(key);
      return { ...prev, selected };
    });
  };

  // 確認後の実削除 → Absence 検証結果を反映
  const handleForgetConfirm = async () => {
    if (!forgetState) return;
    const keys = [...forgetState.selected];
    if (keys.length === 0) return;
    setForgetting(true);
    setError(null);
    try {
      const res = await forgetConfirm(userId, keys);
      setForgetState(null);
      if (res.verified) {
        setNotice(`${res.deleted_count}件の記憶を忘れました。`);
      } else {
        const leaked = res.leaked_keys.length;
        setNotice(
          `${res.deleted_count}件を削除しましたが、${leaked}件の記憶が残っている可能性があります。検索で残存を確認し、必要なら個別に削除してください。`,
        );
      }
      await load(activeQueryRef.current);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setForgetting(false);
    }
  };

  // モーダル表示中は Escape で閉じる (削除中は閉じない)
  useEffect(() => {
    if (!forgetState) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !forgetting) setForgetState(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [forgetState, forgetting]);

  return (
    <aside className="memory-panel">
      <div className="memory-header">
        <h2>AIの記憶</h2>
        <p className="memory-note">
          バックグラウンドの記憶統合には30秒ほどかかるため、最新の会話の内容は少し遅れて反映されます。
        </p>
        <div className="memory-search">
          <input
            type="text"
            value={query}
            placeholder="記憶を検索..."
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.nativeEvent.isComposing) {
                handleSearch();
              }
            }}
          />
          <button type="button" className="btn" onClick={handleSearch}>
            検索
          </button>
        </div>
        <button
          type="button"
          className="btn memory-forget-all"
          disabled={!query.trim() || previewing}
          title="この検索語に関連する記憶をまとめて忘れる"
          onClick={() => void handleForgetPreview()}
        >
          {previewing ? "候補を取得中..." : "この検索結果をまとめて忘れる"}
        </button>
      </div>
      {error && <div className="panel-error">{error}</div>}
      {notice && (
        <div className="memory-notice" onClick={() => setNotice(null)}>
          {notice}
        </div>
      )}
      <div className="memory-list">
        {loading && memories.length === 0 && (
          <div className="sidebar-note">読み込み中...</div>
        )}
        {!loading && memories.length === 0 && (
          <div className="sidebar-note">記憶はまだありません</div>
        )}
        {memories.map((m) => (
          <div key={m.key} className="memory-item">
            <div className="memory-content">{m.content}</div>
            <div className="memory-meta">
              <span>{formatDate(m.updated_at)}</span>
              {m.score !== null && (
                <span className="memory-score">score: {m.score.toFixed(2)}</span>
              )}
              <button
                type="button"
                className="btn memory-delete"
                title="この記憶を削除"
                onClick={() => void handleDelete(m.key)}
              >
                削除
              </button>
            </div>
          </div>
        ))}
      </div>

      {forgetState && (
        <div
          className="modal-overlay"
          onClick={() => !forgetting && setForgetState(null)}
        >
          <div
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="forget-modal-title"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="modal-title" id="forget-modal-title">
              記憶をまとめて忘れる
            </h3>
            <p className="modal-note">
              「{forgetState.query}」に関連する次の記憶を削除します。チェックを外すと残せます。
              <strong>この操作は取り消せません。</strong>
            </p>
            <div className="modal-list">
              {forgetState.candidates.map((c) => (
                <label key={c.key} className="modal-item">
                  <input
                    type="checkbox"
                    checked={forgetState.selected.has(c.key)}
                    onChange={() => toggleSelected(c.key)}
                  />
                  <span className="memory-content">{c.content}</span>
                </label>
              ))}
            </div>
            <div className="modal-actions">
              <button
                type="button"
                className="btn"
                disabled={forgetting}
                autoFocus
                onClick={() => setForgetState(null)}
              >
                キャンセル
              </button>
              <button
                type="button"
                className="btn modal-danger"
                disabled={forgetting || forgetState.selected.size === 0}
                onClick={() => void handleForgetConfirm()}
              >
                {forgetting
                  ? "削除中..."
                  : `${forgetState.selected.size}件を忘れる`}
              </button>
            </div>
          </div>
        </div>
      )}
    </aside>
  );
}
