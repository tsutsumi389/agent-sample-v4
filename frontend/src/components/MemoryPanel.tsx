import { useCallback, useEffect, useRef, useState } from "react";
import { deleteMemory, listMemories } from "../api/client";
import type { Memory } from "../api/types";

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

export function MemoryPanel({ userId, refreshKey }: Props) {
  const [memories, setMemories] = useState<Memory[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const activeQueryRef = useRef("");

  const load = useCallback(
    async (q: string) => {
      setLoading(true);
      setError(null);
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
      </div>
      {error && <div className="panel-error">{error}</div>}
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
    </aside>
  );
}
