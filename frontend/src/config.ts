export const API_BASE = import.meta.env.VITE_API_BASE ?? "/api/v1";

// When set (e.g. "/models"), the TF.js detectors load from bundled files under
// public/ instead of the Google CDN. Populate with scripts/download-models.sh.
export const MODEL_BASE = import.meta.env.VITE_MODEL_BASE ?? "";

// Seconds a "hand at eye level" pose must be held to trigger capture.
export const POSE_HOLD_MS = 2000;

// Seconds the result screen stays up before returning to the attract loop.
export const RESULT_TIMEOUT_MS = 40000;

// Keys (from a wireless button that presents as a keyboard) that trigger capture.
export const CAPTURE_KEYS = new Set([" ", "Enter", "w", "W"]);
