import { useMemo, useState } from "react";

import type { AnalyzeOk, LabelInfo } from "../lib/api";
import { sendFeedback } from "../lib/api";

function pct(x: number): string {
  return `${Math.round(x * 100)}%`;
}

export function ResultScreen({
  result,
  labels,
  onDone,
}: {
  result: AnalyzeOk;
  labels: LabelInfo[];
  onDone: () => void;
}) {
  const byName = useMemo(() => {
    const m = new Map<string, LabelInfo>();
    for (const l of labels) m.set(l.name, l);
    return m;
  }, [labels]);

  const [fb, setFb] = useState<"idle" | "picking" | "sent">("idle");

  const info = (name: string) => byName.get(name);
  const top = result.top;
  const topInfo = info(top.name);
  const accent = top.accent ?? topInfo?.accent ?? "#c65d97";
  const title = top.display_name ?? topInfo?.display_name ?? top.name;
  const top3 = result.ranked.slice(0, 3);

  const submit = async (label: string) => {
    setFb("sent");
    await sendFeedback(result.record_id, label);
  };

  return (
    <div className="result" style={{ ["--accent" as string]: accent }}>
      {result.display_overlay_b64 && (
        <img
          className="result-img"
          src={`data:image/jpeg;base64,${result.display_overlay_b64}`}
          alt="segmented outfit"
        />
      )}

      <div className="result-body">
        <p className="eyebrow">The Fashion Police say</p>
        <h1 className="verdict">{title}</h1>
        {result.uncertain && <p className="uncertain">…though this one's a close call.</p>}
        {topInfo?.blurb && <p className="blurb">{topInfo.blurb}</p>}

        <ul className="bars">
          {top3.map((s) => {
            const li = info(s.name);
            return (
              <li key={s.name}>
                <span className="bar-name">{s.display_name ?? li?.display_name ?? s.name}</span>
                <span className="bar-track">
                  <span
                    className="bar-fill"
                    style={{ width: pct(s.score), background: s.accent ?? li?.accent ?? "#888" }}
                  />
                </span>
                <span className="bar-val">{pct(s.score)}</span>
              </li>
            );
          })}
        </ul>

        <div className="feedback">
          {fb === "idle" && (
            <>
              <span>Not you?</span>
              <button className="link" onClick={() => setFb("picking")}>
                Set it straight
              </button>
            </>
          )}
          {fb === "picking" && (
            <div className="grid">
              {labels.map((l) => (
                <button key={l.name} className="chip" onClick={() => submit(l.name)}>
                  {l.display_name}
                </button>
              ))}
            </div>
          )}
          {fb === "sent" && <span>Noted — thanks.</span>}
        </div>

        <p className="disclaimer">
          This is a guess based on your clothes, made for fun and for provoking a
          conversation about automated judgement. It is not a verdict.
        </p>

        <button className="btn ghost" onClick={onDone}>
          Done
        </button>
      </div>
    </div>
  );
}
