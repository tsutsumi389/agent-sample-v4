import { useState } from "react";
import { z } from "zod";
import type { UIResource as UIResourceData } from "../api/types";
import { ToolCallChip } from "./ToolCallChip";

/**
 * 生成的UI(チャット内GUI)の宣言的レンダラ。
 *
 * バックエンドのUIツールが返した封筒 (component 名 + props) を、フロント側の
 * ホワイトリスト REGISTRY で解決して本物のReactコンポーネントとして描く。
 * - 任意HTML/JSは一切実行しない (XSS面ゼロ)。
 * - props は各 View 内で Zod 検証してから使う (MCP/LLM由来の不正データを弾く)。
 * - 未知 component / 検証失敗時は ToolCallChip にフォールバック (前方互換)。
 *
 * 設計書: docs/guide/11-generative-ui.html
 */

/** UI上の操作をエージェントへ環流するためのアクション。ChatView が文字列整形して再送する。 */
export interface UIAction {
  uiId: string;
  component: string;
  action: string;
  payload: Record<string, unknown>;
}

interface ViewProps {
  props: unknown;
  /** アクションを環流する。送信が受理されたら true、ビジー等で破棄されたら false。 */
  onAction: (a: Omit<UIAction, "uiId" | "component">) => boolean;
}

// ---- table ----
const tableSchema = z.object({
  title: z.string().optional(),
  columns: z.array(z.string()),
  rows: z.array(z.array(z.union([z.string(), z.number()]))),
});

function TableView({ props }: ViewProps) {
  const r = tableSchema.safeParse(props);
  if (!r.success) return <FallbackNote reason="table の props 形式が不正です" />;
  const { title, columns, rows } = r.data;
  return (
    <div className="ui-table">
      {title && <div className="ui-title">{title}</div>}
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {columns.map((c, i) => (
                <th key={i}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, ri) => (
              <tr key={ri}>
                {row.map((cell, ci) => (
                  <td key={ci}>{String(cell)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---- chart (シンプルな棒グラフ。外部ライブラリ不使用) ----
const chartSchema = z
  .object({
    title: z.string().optional(),
    labels: z.array(z.string()),
    values: z.array(z.number()),
    unit: z.string().optional().default(""),
  })
  // labels と values の長さが揃わないとラベルがずれるため弾く。
  .refine((d) => d.labels.length === d.values.length, {
    message: "labels と values の長さが一致しません",
  });

function ChartView({ props }: ViewProps) {
  const r = chartSchema.safeParse(props);
  if (!r.success) return <FallbackNote reason="chart の props 形式が不正です" />;
  const { title, labels, values, unit } = r.data;
  const max = Math.max(1, ...values);
  return (
    <div className="ui-chart">
      {title && <div className="ui-title">{title}</div>}
      <div className="ui-chart-bars">
        {values.map((v, i) => (
          <div className="ui-chart-row" key={i}>
            <span className="ui-chart-label">{labels[i] ?? i}</span>
            <span className="ui-chart-track">
              <span
                className="ui-chart-fill"
                style={{ width: `${(v / max) * 100}%` }}
              />
            </span>
            <span className="ui-chart-value">
              {v}
              {unit}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---- card ----
const cardSchema = z.object({
  title: z.string(),
  body: z.string().optional().default(""),
  fields: z
    .array(z.object({ label: z.string(), value: z.string() }))
    .optional()
    .default([]),
});

function CardView({ props }: ViewProps) {
  const r = cardSchema.safeParse(props);
  if (!r.success) return <FallbackNote reason="card の props 形式が不正です" />;
  const { title, body, fields } = r.data;
  return (
    <div className="ui-card">
      <div className="ui-card-title">{title}</div>
      {body && <p className="ui-card-body">{body}</p>}
      {fields.length > 0 && (
        <dl className="ui-card-fields">
          {fields.map((f, i) => (
            <div className="ui-card-field" key={i}>
              <dt>{f.label}</dt>
              <dd>{f.value}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}

// ---- form ----
const formSchema = z.object({
  title: z.string().optional(),
  submit_label: z.string().optional().default("送信"),
  fields: z.array(
    z.object({
      name: z.string(),
      label: z.string().optional(),
      type: z.enum(["text", "number", "textarea"]).optional().default("text"),
      placeholder: z.string().optional(),
    }),
  ),
});

function FormView({ props, onAction }: ViewProps) {
  const r = formSchema.safeParse(props);
  const [values, setValues] = useState<Record<string, string>>({});
  const [submitted, setSubmitted] = useState(false);
  if (!r.success) return <FallbackNote reason="form の props 形式が不正です" />;
  const { title, submit_label, fields } = r.data;

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (submitted) return;
    // number 項目は数値へ変換して環流する (空欄は除外)。
    const payload: Record<string, unknown> = {};
    for (const f of fields) {
      const raw = values[f.name];
      if (raw === undefined || raw === "") continue;
      payload[f.name] = f.type === "number" && raw !== "" ? Number(raw) : raw;
    }
    // ストリーミング中等で送信が破棄された場合はフォームを編集可能のまま残し、
    // 受理された場合のみ送信済みにロックする (サイレントな送信消失を防ぐ)。
    const accepted = onAction({ action: "submit", payload });
    if (accepted) setSubmitted(true);
  };

  return (
    <form className="ui-form" onSubmit={submit}>
      {title && <div className="ui-title">{title}</div>}
      {fields.map((f) => (
        <label className="ui-form-field" key={f.name}>
          <span>{f.label ?? f.name}</span>
          {f.type === "textarea" ? (
            <textarea
              className="input ui-form-control"
              value={values[f.name] ?? ""}
              placeholder={f.placeholder}
              disabled={submitted}
              onChange={(e) =>
                setValues((v) => ({ ...v, [f.name]: e.target.value }))
              }
            />
          ) : (
            <input
              className="input ui-form-control"
              type={f.type === "number" ? "number" : "text"}
              value={values[f.name] ?? ""}
              placeholder={f.placeholder}
              disabled={submitted}
              onChange={(e) =>
                setValues((v) => ({ ...v, [f.name]: e.target.value }))
              }
            />
          )}
        </label>
      ))}
      <button type="submit" className="btn send ui-form-submit" disabled={submitted}>
        {submitted ? "送信済み" : submit_label}
      </button>
    </form>
  );
}

function FallbackNote({ reason }: { reason: string }) {
  return <div className="ui-fallback">表示できません: {reason}</div>;
}

const REGISTRY: Record<string, React.FC<ViewProps>> = {
  table: TableView,
  chart: ChartView,
  card: CardView,
  form: FormView,
};

interface Props {
  res: UIResourceData;
  onAction: (a: UIAction) => boolean;
}

export function UIResource({ res, onAction }: Props) {
  // (B) サンドボックスiframe方式は将来。現状は宣言的方式のみ描画する。
  if (res.mode === "iframe") {
    return (
      <ToolCallChip
        call={{ id: res.id, name: res.name ?? res.component, args: {} }}
      />
    );
  }
  const Component = REGISTRY[res.component];
  if (!Component) {
    // 未知 component はフォールバック (前方互換 = コア無改修のフロント版)。
    return (
      <ToolCallChip
        call={{ id: res.id, name: res.name ?? res.component, args: {} }}
      />
    );
  }
  return (
    <div className="ui-resource">
      <Component
        props={res.props}
        onAction={(a) =>
          onAction({ uiId: res.id, component: res.component, ...a })
        }
      />
    </div>
  );
}
