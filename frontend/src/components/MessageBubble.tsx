import { ToolCallChip, type UIToolCall } from "./ToolCallChip";

export interface UIMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  toolCalls: UIToolCall[];
}

interface Props {
  message: UIMessage;
  streaming?: boolean;
}

export function MessageBubble({ message, streaming = false }: Props) {
  return (
    <div className={`message-row ${message.role}`}>
      <div className={`message-bubble ${message.role}`}>
        <div className="message-role">
          {message.role === "user" ? "あなた" : "アシスタント"}
        </div>
        {message.toolCalls.length > 0 && (
          <div className="tool-chips">
            {message.toolCalls.map((call) => (
              <ToolCallChip key={call.id} call={call} />
            ))}
          </div>
        )}
        {(message.content !== "" || message.role === "user") && (
          <div className="message-content">{message.content}</div>
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
