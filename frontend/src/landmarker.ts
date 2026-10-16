/**
 * MediaPipe Hand Landmarker, self-hosted.
 *
 * The wasm bundle and the .task model are served from /mediapipe/ rather than a
 * CDN, so a demo does not depend on the network (CLAUDE.md 14, pitfall 6).
 * Run `npm run vendor` to put them there.
 */

import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

const WASM_PATH = "/mediapipe/wasm";
const MODEL_PATH = "/mediapipe/hand_landmarker.task";

/** A hand, ready to send over the wire. */
export interface DetectedHand {
  /** 21 landmarks as [x, y, z], normalized the way MediaPipe reports them. */
  readonly landmarks: [number, number, number][];
  /** The physical hand, corrected for MediaPipe's selfie assumption. */
  readonly handedness: "Left" | "Right";
}

export class LandmarkerError extends Error {}

export async function createLandmarker(): Promise<HandLandmarker> {
  try {
    const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);
    return await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_PATH, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
  } catch (cause) {
    throw new LandmarkerError(
      "Could not load the hand model. Have you run `npm run vendor`?",
      { cause },
    );
  }
}

/**
 * Detect one hand in the current video frame, or null if there is none.
 *
 * `timestampMs` must increase between calls: VIDEO mode uses it for tracking
 * and rejects a timestamp that goes backwards.
 */
export function detectHand(
  landmarker: HandLandmarker,
  video: HTMLVideoElement,
  timestampMs: number,
): DetectedHand | null {
  const result = landmarker.detectForVideo(video, timestampMs);
  const landmarks = result.landmarks[0];
  if (!landmarks || landmarks.length !== 21) return null;

  return {
    landmarks: landmarks.map(({ x, y, z }) => [x, y, z] as [number, number, number]),
    handedness: physicalHand(result.handedness[0]?.[0]?.categoryName),
  };
}

/**
 * Undo MediaPipe's selfie assumption.
 *
 * Its documentation: "handedness is determined assuming the input image is
 * mirrored, i.e., taken with a front-facing/selfie camera with images flipped
 * horizontally. If it is not the case, please swap the handedness output in the
 * application." getUserMedia hands us the unflipped frame -- mirroring the
 * preview in CSS does not change the pixels MediaPipe reads -- so the label
 * arrives inverted and is swapped here. The backend is then told the physical
 * hand with `mirrored: false` (CLAUDE.md 7).
 */
function physicalHand(reported: string | undefined): "Left" | "Right" {
  return reported === "Left" ? "Right" : "Left";
}
