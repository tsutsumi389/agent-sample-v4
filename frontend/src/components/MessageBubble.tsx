import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ToolCallChip, type UIToolCall } from "./ToolCallChip";
import { UIResource, type UIAction } from "./UIResource";

export interface UIMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  toolCalls: UIToolCall[];
}

interface Props {
  message: UIMessage;
  streaming?: boolean;
  /** 生成的UI上の操作 (フォーム送信等) をエージェントへ環流する。受理されたら true。 */
  onUIAction?: (a: UIAction) => boolean;
}

export function MessageBubble({ message, streaming = false, onUIAction }: Props) {
  // UI 封筒を持つツール呼び出しは GUI として展開描画し、それ以外はチップ表示。
  const uiCalls = message.toolCalls.filter((c) => c.ui);
  const chipCalls = message.toolCalls.filter((c) => !c.ui);
  return (
    <div className={`message-row ${message.role}`}>
      <div className={`message-bubble ${message.role}`}>
        <div className="message-role">
          {message.role === "user" ? "あなた" : "アシスタント"}
        </div>
        {chipCalls.length > 0 && (
          <div className="tool-chips">
            {chipCalls.map((call) => (
              <ToolCallChip key={call.id} call={call} />
            ))}
          </div>
        )}
        {uiCalls.length > 0 && (
          <div className="ui-resources">
            {uiCalls.map((call) => (
              <UIResource
                key={call.id}
                res={call.ui!}
                onAction={(a) => onUIAction?.(a) ?? false}
              />
            ))}
          </div>
        )}
        {(message.content !== "" || message.role === "user") && (
          <div className={`message-content ${message.role === "assistant" ? "markdown" : ""}`}>
            {message.role === "assistant" ? (
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
            ) : (
              message.content
            )}
          </div>
        )}
        {streaming && message.content === "" && message.toolCalls.length === 0 && (
          <div className="message-content thinking">考え中...</div>
        )}
        {streaming && (message.content !== "" || message.toolCalls.length > 0) && (
          <span className="cursor" />
        )}
      </div>
    </div>
  );
}
