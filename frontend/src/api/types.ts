// api_contract.md と厳密に対応する型定義。

export interface Thread {
  thread_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
}

// 生成的UI: ツールが返した UI 封筒 (backend coerce_ui の結果)。
export interface UIResource {
  id: string;
  name?: string;
  component: string;
  mode: "declarative" | "iframe";
  props: Record<string, unknown>;
  v: number;
}

export interface MessageOut {
  role: "user" | "assistant" | "tool";
  content: string;
  id: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  name?: string;
  ui?: UIResource;
}

export interface Memory {
  key: string;
  content: string;
  namespace: string[];
  updated_at: string;
  score: number | null;
}

export interface ToolInfo {
  name: string;
  description: string;
  source: string;
}

// --- スコープ付き一括忘却 (POST /api/memory/forget/*) ---

export interface ForgetCandidate {
  key: string;
  content: string;
  score: number | null;
  updated_at: string | null;
}

export interface ForgetPreviewResult {
  candidates: ForgetCandidate[];
}

export interface ForgetConfirmResult {
  deleted_count: number;
  verified: boolean;
  leaked_keys: string[];
}

// --- SSE イベント (POST /api/chat/stream) ---

export interface TokenEvent {
  content: string;
  node: string;
}

export interface ToolCallEvent {
  id: string;
  name: string;
  args: Record<string, unknown>;
}

export interface ToolResultEvent {
  id: string;
  name: string;
  content: string;
}

// SSE ui_resource イベントの data。UIResource と同形。
export type UIResourceEvent = UIResource;

export interface ProgressEvent {
  status: string;
}

export interface DoneEvent {
  thread_id: string;
  title: string;
}

export interface ErrorEvent {
  message: string;
}

export interface ChatStreamRequest {
  message: string;
  thread_id: string;
  user_id: string;
}
