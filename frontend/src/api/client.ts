import type { Memory, MessageOut, Thread, ToolInfo } from "./types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = (await res.json()) as { detail?: string };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // JSON でないエラーボディは無視
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export function listThreads(userId: string): Promise<{ threads: Thread[] }> {
  return request(`/api/threads?user_id=${encodeURIComponent(userId)}`);
}

export function createThread(
  userId: string,
  title: string | null = null,
): Promise<Thread> {
  return request("/api/threads", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, title }),
  });
}

export function getMessages(
  threadId: string,
  userId: string,
): Promise<{ thread_id: string; messages: MessageOut[] }> {
  return request(
    `/api/threads/${encodeURIComponent(threadId)}/messages?user_id=${encodeURIComponent(userId)}`,
  );
}

export function deleteThread(
  threadId: string,
  userId: string,
): Promise<{ deleted: boolean }> {
  return request(
    `/api/threads/${encodeURIComponent(threadId)}?user_id=${encodeURIComponent(userId)}`,
    { method: "DELETE" },
  );
}

export function listMemories(
  userId: string,
  query?: string,
  limit = 20,
): Promise<{ user_id: string; memories: Memory[] }> {
  const params = new URLSearchParams({ user_id: userId, limit: String(limit) });
  if (query) params.set("query", query);
  return request(`/api/memory?${params.toString()}`);
}

export function deleteMemory(
  key: string,
  userId: string,
): Promise<{ deleted: boolean }> {
  return request(
    `/api/memory/${encodeURIComponent(key)}?user_id=${encodeURIComponent(userId)}`,
    { method: "DELETE" },
  );
}

export function listTools(): Promise<{ tools: ToolInfo[] }> {
  return request("/api/tools");
}
