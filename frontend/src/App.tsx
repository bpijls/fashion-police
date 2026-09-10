import { useCallback, useEffect, useReducer, useRef, useState } from "react";

import { CAPTURE_KEYS, POSE_HOLD_MS, RESULT_TIMEOUT_MS } from "./config";
import { initialState, reducer } from "./kiosk";
import { analyze, getLabels, type LabelInfo } from "./lib/api";
import { grabFrame, startCamera, stopStream } from "./lib/camera";
import { canvasToJpegBlob, redactAndCrop } from "./lib/redact";
import {
  detectFaces,
  estimatePose,
  hasPerson,
  isCapturePose,
  loadDetectors,
  type Keypoint,
} from "./lib/vision";
import { ResultScreen } from "./components/ResultScreen";

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [labels, setLabels] = useState<LabelInfo[]>([]);
  const [holdPct, setHoldPct] = useState(0);

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const lastPoseRef = useRef<Keypoint[]>([]);
  const holdStartRef = useRef<number | null>(null);
  const phaseRef = useRef(state.phase);
  const busyRef = useRef(false);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    phaseRef.current = state.phase;
  }, [state.phase]);

  // --- capture -> redact (in-browser) -> send ---------------------------
  const doCapture = useCallback(async () => {
    const video = videoRef.current;
    if (!video || busyRef.current) return;
    if (phaseRef.current !== "idle" && phaseRef.current !== "ready") return;
    busyRef.current = true;
    holdStartRef.current = null;
    setHoldPct(0);

    try {
      const frame = grabFrame(video); // raw frame — never stored anywhere else
      const faces = await detectFaces(frame);
      const { canvas } = redactAndCrop(frame, faces, lastPoseRef.current);
      const preview = canvas.toDataURL("image/jpeg", 0.85);
      dispatch({ type: "capture_started", preview });

      const blob = await canvasToJpegBlob(canvas, 0.9);
      // give the "anonymising" copy a beat to be read
      await new Promise((r) => setTimeout(r, 650));
      dispatch({ type: "sending" });

      const res = await analyze(blob);
      if (res.kind === "ok") dispatch({ type: "analysed", result: res });
      else if (res.kind === "face_visible") dispatch({ type: "face_visible", message: res.message });
      else dispatch({ type: "failed", message: res.message });
    } catch (err) {
      dispatch({ type: "failed", message: err instanceof Error ? err.message : "capture failed" });
    } finally {
      busyRef.current = false;
    }
  }, []);

  // --- detection loop --------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    let frame = 0;

    const tick = async () => {
      if (cancelled) return;
      const video = videoRef.current;
      const phase = phaseRef.current;
      const scanning = phase === "idle" || phase === "ready";

      if (video && video.readyState >= 2 && scanning && !busyRef.current && frame++ % 2 === 0) {
        try {
          const pose = await estimatePose(video);
          lastPoseRef.current = pose;
          const present = hasPerson(pose);

          if (phase === "idle" && present) {
            const faces = await detectFaces(video);
            if (faces.length > 0) dispatch({ type: "person_present" });
          } else if (phase === "ready") {
            if (!present) {
              dispatch({ type: "person_absent" });
            } else if (isCapturePose(pose)) {
              if (holdStartRef.current === null) holdStartRef.current = performance.now();
              const held = performance.now() - holdStartRef.current;
              setHoldPct(Math.min(1, held / POSE_HOLD_MS));
              if (held >= POSE_HOLD_MS) void doCapture();
            } else {
              holdStartRef.current = null;
              setHoldPct(0);
            }
          }
        } catch {
          /* transient detector hiccup — keep looping */
        }
      }
      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      cancelAnimationFrame(rafRef.current);
    };
  }, [doCapture]);

  // --- keyboard / wireless button ------------------------------------
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (CAPTURE_KEYS.has(e.key)) {
        e.preventDefault();
        if (phaseRef.current === "result" || phaseRef.current === "error") {
          dispatch({ type: "reset" });
        } else {
          void doCapture();
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [doCapture]);

  // --- result auto-return -------------------------------------------
  useEffect(() => {
    if (state.phase !== "result" && state.phase !== "error") return;
    const ms = state.phase === "result" ? RESULT_TIMEOUT_MS : 6000;
    const t = setTimeout(() => dispatch({ type: "reset" }), ms);
    return () => clearTimeout(t);
  }, [state.phase]);

  // --- taxonomy for the result + feedback screens ------------------
  useEffect(() => {
    let alive = true;
    getLabels()
      .then((l) => alive && setLabels(l))
      .catch((e) => console.warn("could not load labels:", e));
    return () => {
      alive = false;
    };
  }, []);

  // --- startup: detectors + camera --------------------------------
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        await loadDetectors();
        const stream = await startCamera();
        if (cancelled) {
          stopStream(stream);
          return;
        }
        streamRef.current = stream;
        const video = videoRef.current!;
        video.srcObject = stream;
        await video.play();
        dispatch({ type: "ready_to_start" });
      } catch (err) {
        dispatch({
          type: "failed",
          message:
            err instanceof Error && err.name === "NotAllowedError"
              ? "Camera permission is needed for this installation."
              : "Could not start the camera.",
        });
      }
    })();

    return () => {
      cancelled = true;
      stopStream(streamRef.current);
    };
  }, []);

  // keep the screen awake
  useEffect(() => {
    let lock: WakeLockSentinel | null = null;
    const req = () => navigator.wakeLock?.request("screen").then((l) => (lock = l)).catch(() => {});
    void req();
    const onVis = () => document.visibilityState === "visible" && void req();
    document.addEventListener("visibilitychange", onVis);
    return () => {
      document.removeEventListener("visibilitychange", onVis);
      void lock?.release();
    };
  }, []);

  return (
    <div className="stage" data-phase={state.phase}>
      <video ref={videoRef} className="mirror" playsInline muted />
      <div className="scrim" />

      {state.phase === "loading" && <Centre title="Starting up" sub="One moment…" />}

      {state.phase === "idle" && (
        <Centre title="Fashion Police" sub="Step in front of the camera" />
      )}

      {state.phase === "ready" && (
        <div className="centre">
          <HoldRing pct={holdPct} />
          <h1>{holdPct > 0 ? "Hold still…" : "Ready"}</h1>
          <p>{state.hint ?? "Raise a hand to your eye and hold — or press the button."}</p>
        </div>
      )}

      {state.phase === "redacting" && (
        <div className="centre">
          {state.redactedPreview && (
            <img className="preview" src={state.redactedPreview} alt="anonymised capture" />
          )}
          <p className="steps">
            Anonymising&nbsp;✓&nbsp;&nbsp;→&nbsp;&nbsp;Sending anonymised image&nbsp;✓
          </p>
        </div>
      )}

      {state.phase === "analysing" && (
        <div className="centre">
          {state.redactedPreview && (
            <img className="preview" src={state.redactedPreview} alt="anonymised capture" />
          )}
          <Spinner />
          <p>Reading the outfit…</p>
        </div>
      )}

      {state.phase === "result" && state.result && (
        <ResultScreen result={state.result} labels={labels} onDone={() => dispatch({ type: "reset" })} />
      )}

      {state.phase === "error" && (
        <div className="centre">
          <h1>Just a moment</h1>
          <p>{state.error}</p>
          <button className="btn" onClick={() => dispatch({ type: "reset" })}>
            Try again
          </button>
        </div>
      )}
    </div>
  );
}

function Centre({ title, sub }: { title: string; sub: string }) {
  return (
    <div className="centre">
      <h1>{title}</h1>
      <p>{sub}</p>
    </div>
  );
}

function Spinner() {
  return <div className="spinner" aria-label="working" />;
}

function HoldRing({ pct }: { pct: number }) {
  const r = 54;
  const c = 2 * Math.PI * r;
  return (
    <svg className="ring" width="140" height="140" viewBox="0 0 140 140" aria-hidden="true">
      <circle cx="70" cy="70" r={r} className="ring-track" />
      <circle
        cx="70"
        cy="70"
        r={r}
        className="ring-fill"
        strokeDasharray={c}
        strokeDashoffset={c * (1 - pct)}
      />
    </svg>
  );
}
