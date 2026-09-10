import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-backend-cpu";
import * as tf from "@tensorflow/tfjs-core";
import * as blazeface from "@tensorflow-models/blazeface";
import * as poseDetection from "@tensorflow-models/pose-detection";

import { MODEL_BASE } from "../config";

export type Keypoint = poseDetection.Keypoint;
export interface FaceBox {
  x: number;
  y: number;
  w: number;
  h: number;
  score: number;
}
export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

let poseDetector: poseDetection.PoseDetector | null = null;
let faceModel: blazeface.BlazeFaceModel | null = null;

export async function loadDetectors(): Promise<void> {
  try {
    await tf.setBackend("webgl");
    await tf.ready();
  } catch {
    await tf.setBackend("cpu");
    await tf.ready();
  }

  const moveNetCfg: poseDetection.MoveNetModelConfig = {
    modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
  };
  if (MODEL_BASE) moveNetCfg.modelUrl = `${MODEL_BASE}/movenet/model.json`;

  const faceCfg: Parameters<typeof blazeface.load>[0] = MODEL_BASE
    ? { modelUrl: `${MODEL_BASE}/blazeface/model.json` }
    : undefined;

  [poseDetector, faceModel] = await Promise.all([
    poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, moveNetCfg),
    blazeface.load(faceCfg),
  ]);
}

function kp(kpts: Keypoint[], name: string): Keypoint | undefined {
  const k = kpts.find((p) => p.name === name);
  return k && (k.score ?? 0) > 0.3 ? k : undefined;
}

export async function estimatePose(
  source: HTMLVideoElement | HTMLCanvasElement,
): Promise<Keypoint[]> {
  if (!poseDetector) return [];
  const poses = await poseDetector.estimatePoses(source, { flipHorizontal: false });
  return poses[0]?.keypoints ?? [];
}

export async function detectFaces(
  source: HTMLVideoElement | HTMLCanvasElement,
): Promise<FaceBox[]> {
  if (!faceModel) return [];
  const preds = await faceModel.estimateFaces(source, false);
  return preds.map((p) => {
    const [x1, y1] = p.topLeft as [number, number];
    const [x2, y2] = p.bottomRight as [number, number];
    const prob = Array.isArray(p.probability) ? p.probability[0] : (p.probability as number);
    return { x: x1, y: y1, w: x2 - x1, h: y2 - y1, score: prob ?? 1 };
  });
}

/** "Hand at eye level" — hold this to trigger a hands-free capture.
 *  Ported from responsibleIT/fashion-police static/js/camera.js. */
export function isCapturePose(kpts: Keypoint[]): boolean {
  const lw = kp(kpts, "left_wrist");
  const rw = kp(kpts, "right_wrist");
  if (!lw && !rw) return false;

  const le = kp(kpts, "left_eye");
  const re = kp(kpts, "right_eye");
  const nose = kp(kpts, "nose");
  let eyeY: number;
  if (le && re) eyeY = (le.y + re.y) / 2;
  else if (nose) eyeY = nose.y - 20;
  else return false;

  const top = eyeY - 60;
  const bottom = eyeY + 100;
  const atLevel = (w?: Keypoint) => !!w && w.y >= top && w.y <= bottom;
  return atLevel(lw) || atLevel(rw);
}

export function hasPerson(kpts: Keypoint[]): boolean {
  const torso = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"];
  return torso.filter((n) => kp(kpts, n)).length >= 2;
}

function boundOf(points: { x: number; y: number }[]): Rect | null {
  if (points.length === 0) return null;
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const x = Math.min(...xs);
  const y = Math.min(...ys);
  return { x, y, w: Math.max(...xs) - x, h: Math.max(...ys) - y };
}

/** Padded bounding box of the person, clamped to the frame. */
export function personBox(kpts: Keypoint[], frameW: number, frameH: number): Rect | null {
  const confident = kpts.filter((p) => (p.score ?? 0) > 0.3);
  const b = boundOf(confident);
  if (!b) return null;
  const padTop = b.h * 0.28; // keypoints stop at the eyes; leave room for the head
  const padX = b.w * 0.12;
  const padBottom = b.h * 0.05;
  const x = Math.max(0, b.x - padX);
  const y = Math.max(0, b.y - padTop);
  const w = Math.min(frameW - x, b.w + 2 * padX);
  const h = Math.min(frameH - y, b.h + padTop + padBottom);
  return { x, y, w, h };
}

/** Coarse head box from face keypoints — the fail-closed redaction fallback
 *  when face detection misses. */
export function headBox(kpts: Keypoint[]): Rect | null {
  const head = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
    .map((n) => kp(kpts, n))
    .filter((k): k is Keypoint => !!k);
  const b = boundOf(head);
  if (!b) return null;
  const spread = Math.max(b.w, 40);
  return {
    x: b.x - spread * 0.9,
    y: b.y - spread * 1.6,
    w: b.w + spread * 1.8,
    h: b.h + spread * 2.6,
  };
}
