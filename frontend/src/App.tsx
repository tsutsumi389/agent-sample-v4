import { useCallback, useState } from "react";
import type { Thread } from "./api/types";
import { ChatView } from "./components/ChatView";
import { MemoryPanel } from "./components/MemoryPanel";
import { Sidebar } from "./components/Sidebar";
import { useThreads } from "./hooks/useThreads";

const USER_ID: string = import.meta.env.VITE_USER_ID ?? "default-user";

export default function App() {
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [memoryRefreshKey, setMemoryRefreshKey] = useState(0);
  const { threads, loading, refresh, create, remove } = useThreads(USER_ID);

  const handleCreate = useCallback(() => {
    void create()
      .then((t) => setActiveThreadId(t.thread_id))
      .catch(() => undefined);
  }, [create]);

  const handleDelete = useCallback(
    (threadId: string) => {
      void remove(threadId)
        .then(() =>
          setActiveThreadId((cur) => (cur === threadId ? null : cur)),
        )
        .catch(() => undefined);
    },
    [remove],
  );

  // ChatView がスレッド未選択のまま送信 → 新規作成された場合
  const handleThreadCreated = useCallback(
    (thread: Thread) => {
      setActiveThreadId(thread.thread_id);
      void refresh();
    },
    [refresh],
  );

  // done イベント: サイドバーのタイトル更新 + メモリパネル再取得
  const handleDone = useCallback(() => {
    void refresh();
    setMemoryRefreshKey((k) => k + 1);
  }, [refresh]);

  return (
    <div className="app">
      <Sidebar
        threads={threads}
        activeThreadId={activeThreadId}
        loading={loading}
        onSelect={setActiveThreadId}
        onCreate={handleCreate}
        onDelete={handleDelete}
      />
      <ChatView
        threadId={activeThreadId}
        userId={USER_ID}
        onThreadCreated={handleThreadCreated}
        onDone={handleDone}
      />
      <MemoryPanel userId={USER_ID} refreshKey={memoryRefreshKey} />
    </div>
  );
}
