import { useCallback, useEffect, useState } from "react";
import { createThread, deleteThread, listThreads } from "../api/client";
import type { Thread } from "../api/types";

export function useThreads(userId: string) {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await listThreads(userId);
      setThreads(res.threads);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [userId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const create = useCallback(async (): Promise<Thread> => {
    const thread = await createThread(userId);
    await refresh();
    return thread;
  }, [userId, refresh]);

  const remove = useCallback(
    async (threadId: string): Promise<void> => {
      await deleteThread(threadId, userId);
      await refresh();
    },
    [userId, refresh],
  );

  return { threads, loading, error, refresh, create, remove };
}
