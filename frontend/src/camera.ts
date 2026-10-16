/**
 * Webcam access.
 *
 * getUserMedia only exists in a secure context, so this works on localhost and
 * over HTTPS and nowhere else (CLAUDE.md 14, pitfall 2). The deployed instance
 * therefore needs a certificate before the camera can be demonstrated.
 */

export class CameraError extends Error {}

export interface CameraStream {
  readonly video: HTMLVideoElement;
  /** Real frame size, which is not necessarily what we asked for. */
  readonly width: number;
  readonly height: number;
  stop(): void;
}

const WANTED = { width: { ideal: 1280 }, height: { ideal: 720 } };

export async function startCamera(video: HTMLVideoElement): Promise<CameraStream> {
  if (!window.isSecureContext) {
    throw new CameraError(
      "The camera needs a secure context: open this over https or on localhost.",
    );
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new CameraError("This browser exposes no camera API.");
  }

  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: WANTED, audio: false });
  } catch (cause) {
    throw new CameraError(explain(cause), { cause });
  }

  video.srcObject = stream;
  await video.play();
  await firstFrame(video);

  return {
    video,
    // videoWidth/videoHeight are the decoded frame size, which is what the
    // landmarks are normalized against and therefore what the aspect
    // correction needs (CLAUDE.md 6.2, step 4).
    width: video.videoWidth,
    height: video.videoHeight,
    stop() {
      for (const track of stream.getTracks()) track.stop();
      video.srcObject = null;
    },
  };
}

/** Wait until the frame size is actually known, not just until play() resolves. */
function firstFrame(video: HTMLVideoElement): Promise<void> {
  if (video.videoWidth > 0) return Promise.resolve();
  return new Promise((resolve) => {
    video.addEventListener("loadeddata", () => resolve(), { once: true });
  });
}

function explain(cause: unknown): string {
  const name = cause instanceof DOMException ? cause.name : "";
  switch (name) {
    case "NotAllowedError":
      return "Camera permission was denied. Allow it in the address bar and try again.";
    case "NotFoundError":
      return "No camera was found.";
    case "NotReadableError":
      return "The camera is busy — another application may be using it.";
    default:
      return `The camera could not be started: ${String(cause)}`;
  }
}
