import { useState } from "react";

interface Props {
  streaming: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
}

export function Composer({ streaming, onSend, onStop }: Props) {
  const [text, setText] = useState("");

  const submit = () => {
    const trimmed = text.trim();
    if (trimmed === "" || streaming) return;
    onSend(trimmed);
    setText("");
  };

  return (
    <div className="composer">
      <div className="composer-row">
        <textarea
          className="input composer-input"
          value={text}
          placeholder="メッセージを入力 (Enterで送信 / Shift+Enterで改行)"
          rows={3}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              submit();
            }
          }}
        />
        {streaming ? (
          <button type="button" className="btn stop" onClick={onStop}>
            停止
          </button>
        ) : (
          <button
            type="button"
            className="btn send"
            onClick={submit}
            disabled={text.trim() === ""}
          >
            送信
          </button>
        )}
      </div>
    </div>
  );
}
