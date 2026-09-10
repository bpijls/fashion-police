const DEVICE_KEY = "fp.cameraDeviceId";

export async function listCameras(): Promise<MediaDeviceInfo[]> {
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices.filter((d) => d.kind === "videoinput");
}

export function savedDeviceId(): string | null {
  try {
    return localStorage.getItem(DEVICE_KEY);
  } catch {
    return null;
  }
}

export function saveDeviceId(id: string): void {
  try {
    localStorage.setItem(DEVICE_KEY, id);
  } catch {
    /* private mode — ignore */
  }
}

export async function startCamera(deviceId?: string | null): Promise<MediaStream> {
  const tries: MediaStreamConstraints[] = [];
  if (deviceId) tries.push({ video: { deviceId: { exact: deviceId } }, audio: false });
  tries.push({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  tries.push({ video: true, audio: false });

  let lastErr: unknown;
  for (const constraints of tries) {
    try {
      return await navigator.mediaDevices.getUserMedia(constraints);
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error("camera unavailable");
}

export function stopStream(stream: MediaStream | null): void {
  stream?.getTracks().forEach((t) => t.stop());
}

/** Snapshot the current video frame into a fresh canvas (never stored elsewhere). */
export function grabFrame(video: HTMLVideoElement): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d")!.drawImage(video, 0, 0);
  return canvas;
}
