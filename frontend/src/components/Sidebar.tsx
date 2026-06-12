import type { Thread } from "../api/types";

interface Props {
  threads: Thread[];
  activeThreadId: string | null;
  loading: boolean;
  onSelect: (threadId: string) => void;
  onCreate: () => void;
  onDelete: (threadId: string) => void;
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

export function Sidebar({
  threads,
  activeThreadId,
  loading,
  onSelect,
  onCreate,
  onDelete,
}: Props) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h1 className="app-title">AIエージェント</h1>
        <button type="button" className="btn new-thread" onClick={onCreate}>
          + 新しい会話
        </button>
      </div>
      <nav className="thread-list">
        {loading && threads.length === 0 && (
          <div className="sidebar-note">読み込み中...</div>
        )}
        {!loading && threads.length === 0 && (
          <div className="sidebar-note">会話はまだありません</div>
        )}
        {threads.map((t) => (
          <div
            key={t.thread_id}
            className={`thread-item ${t.thread_id === activeThreadId ? "active" : ""}`}
            onClick={() => onSelect(t.thread_id)}
          >
            <div className="thread-info">
              <div className="thread-title">{t.title}</div>
              <div className="thread-date">{formatDate(t.updated_at)}</div>
            </div>
            <button
              type="button"
              className="btn thread-delete"
              title="スレッドを削除"
              onClick={(e) => {
                e.stopPropagation();
                if (window.confirm("このスレッドを削除しますか？")) {
                  onDelete(t.thread_id);
                }
              }}
            >
              削除
            </button>
          </div>
        ))}
      </nav>
    </aside>
  );
}
