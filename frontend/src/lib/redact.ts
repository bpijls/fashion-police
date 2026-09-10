import type { FaceBox, Keypoint, Rect } from "./vision";
import { headBox, personBox } from "./vision";

function expand(r: Rect, factor: number): Rect {
  const dw = r.w * factor;
  const dh = r.h * factor;
  return { x: r.x - dw / 2, y: r.y - dh / 2, w: r.w + dw, h: r.h + dh };
}

function clampRect(r: Rect, w: number, h: number): Rect {
  const x = Math.max(0, Math.min(r.x, w));
  const y = Math.max(0, Math.min(r.y, h));
  return { x, y, w: Math.min(r.w, w - x), h: Math.min(r.h, h - y) };
}

export interface RedactionResult {
  canvas: HTMLCanvasElement;
  redactedRegions: number;
}

/**
 * Black out every detected/expected face region, then crop to the person.
 * The source frame is only ever read here; nothing leaves this function
 * except the redacted, cropped canvas.
 */
export function redactAndCrop(
  source: HTMLCanvasElement,
  faces: FaceBox[],
  pose: Keypoint[],
): RedactionResult {
  const w = source.width;
  const h = source.height;

  const work = document.createElement("canvas");
  work.width = w;
  work.height = h;
  const ctx = work.getContext("2d")!;
  ctx.drawImage(source, 0, 0);

  const regions: Rect[] = [];
  for (const f of faces) regions.push(expand({ x: f.x, y: f.y, w: f.w, h: f.h }, 0.5));
  const hb = headBox(pose);
  if (hb) regions.push(hb);

  ctx.fillStyle = "#000";
  let painted = 0;
  for (const region of regions) {
    const r = clampRect(region, w, h);
    if (r.w > 1 && r.h > 1) {
      ctx.fillRect(r.x, r.y, r.w, r.h);
      painted++;
    }
  }

  // Crop to the person (data minimisation). Fall back to the full frame.
  const pb = personBox(pose, w, h);
  const crop = pb && pb.w > 40 && pb.h > 40 ? clampRect(pb, w, h) : { x: 0, y: 0, w, h };

  const out = document.createElement("canvas");
  out.width = Math.round(crop.w);
  out.height = Math.round(crop.h);
  out.getContext("2d")!.drawImage(
    work,
    crop.x, crop.y, crop.w, crop.h,
    0, 0, out.width, out.height,
  );

  return { canvas: out, redactedRegions: painted };
}

export function canvasToJpegBlob(canvas: HTMLCanvasElement, quality = 0.9): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error("toBlob failed"))),
      "image/jpeg",
      quality,
    );
  });
}
