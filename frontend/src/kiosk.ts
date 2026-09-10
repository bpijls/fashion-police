import type { AnalyzeOk } from "./lib/api";

export type Phase =
  | "loading"
  | "idle"
  | "ready"
  | "redacting"
  | "analysing"
  | "result"
  | "error";

export interface KioskState {
  phase: Phase;
  hint: string | null;
  redactedPreview: string | null;
  result: AnalyzeOk | null;
  error: string | null;
}

export const initialState: KioskState = {
  phase: "loading",
  hint: null,
  redactedPreview: null,
  result: null,
  error: null,
};

export type KioskAction =
  | { type: "ready_to_start" }
  | { type: "person_present" }
  | { type: "person_absent" }
  | { type: "capture_started"; preview: string }
  | { type: "sending" }
  | { type: "analysed"; result: AnalyzeOk }
  | { type: "face_visible"; message: string }
  | { type: "failed"; message: string }
  | { type: "reset" };

export function reducer(state: KioskState, action: KioskAction): KioskState {
  switch (action.type) {
    case "ready_to_start":
      return { ...initialState, phase: "idle" };
    case "person_present":
      return state.phase === "idle" ? { ...state, phase: "ready", hint: null } : state;
    case "person_absent":
      return state.phase === "ready" ? { ...state, phase: "idle" } : state;
    case "capture_started":
      return { ...state, phase: "redacting", redactedPreview: action.preview, hint: null };
    case "sending":
      return { ...state, phase: "analysing" };
    case "analysed":
      return { ...state, phase: "result", result: action.result };
    case "face_visible":
      return { ...initialState, phase: "ready", hint: action.message };
    case "failed":
      return { ...initialState, phase: "error", error: action.message };
    case "reset":
      return { ...initialState, phase: "idle" };
    default:
      return state;
  }
}
