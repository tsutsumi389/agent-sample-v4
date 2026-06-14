import type { UIResource } from "../api/types";

export interface UIToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
  result?: string;
  /** 生成的UI: このツール呼び出しが返した UI 封筒 (あれば UIResource で描画)。 */
  ui?: UIResource;
}

interface Props {
  call: UIToolCall;
}

export function ToolCallChip({ call }: Props) {
  const hasArgs = Object.keys(call.args).length > 0;
  return (
    <details className="tool-chip">
      <summary>
        <span className="tool-chip-name">{call.name}</span>
        <span
          className={
            call.result !== undefined
              ? "tool-chip-status done"
              : "tool-chip-status running"
          }
        >
          {call.result !== undefined ? "完了" : "実行中"}
        </span>
      </summary>
      <div className="tool-chip-body">
        {hasArgs && (
          <div className="tool-chip-section">
            <div className="tool-chip-label">引数</div>
            <pre>{JSON.stringify(call.args, null, 2)}</pre>
          </div>
        )}
        {call.result !== undefined && (
          <div className="tool-chip-section">
            <div className="tool-chip-label">結果</div>
            <pre>{call.result}</pre>
          </div>
        )}
      </div>
    </details>
  );
}
