import { API_BASE } from "../config";

export interface RankedStyle {
  name: string;
  score: number;
  display_name?: string;
  accent?: string;
}

export interface LabelInfo {
  name: string;
  display_name: string;
  blurb: string;
  accent: string;
}

export interface AnalyzeOk {
  kind: "ok";
  record_id: string;
  top: RankedStyle;
  ranked: RankedStyle[];
  uncertain: boolean;
  display_overlay_b64: string;
}

export type AnalyzeResult =
  | AnalyzeOk
  | { kind: "face_visible"; message: string }
  | { kind: "error"; message: string };

export async function getLabels(): Promise<LabelInfo[]> {
  const r = await fetch(`${API_BASE}/labels`);
  if (!r.ok) throw new Error(`labels ${r.status}`);
  const data = await r.json();
  return data.labels as LabelInfo[];
}

export async function getHealth(): Promise<{ compute: string }> {
  const r = await fetch(`${API_BASE}/health`);
  if (!r.ok) throw new Error(`health ${r.status}`);
  return r.json();
}

export async function analyze(frame: Blob): Promise<AnalyzeResult> {
  const form = new FormData();
  form.append("frame", frame, "frame.jpg");
  let r: Response;
  try {
    r = await fetch(`${API_BASE}/analyze`, { method: "POST", body: form });
  } catch {
    return { kind: "error", message: "Could not reach the server." };
  }
  if (r.status === 422) {
    const body = await r.json().catch(() => ({}));
    return { kind: "face_visible", message: body.message ?? "A face was still visible." };
  }
  if (!r.ok) {
    return { kind: "error", message: `The analysis failed (${r.status}).` };
  }
  const data = await r.json();
  return { kind: "ok", ...data };
}

export async function sendFeedback(recordId: string, correctLabel: string): Promise<boolean> {
  try {
    const r = await fetch(`${API_BASE}/feedback/${encodeURIComponent(recordId)}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ correct_label: correctLabel }),
    });
    return r.ok;
  } catch {
    return false;
  }
}
