/**
 * Hand skeleton drawing, with the fingers a hint mentions picked out.
 *
 * The canvas is mirrored in CSS together with the video, so landmarks are drawn
 * in the coordinates MediaPipe reported and still line up with what the learner
 * sees. Nothing here flips anything.
 */

import type { DetectedHand } from "./landmarker";
import type { Hint } from "./ws-client";

/** Landmark chains, wrist to tip, keyed the way the backend names fingers. */
const FINGER_CHAINS: Record<string, number[]> = {
  thumb: [0, 1, 2, 3, 4],
  index: [0, 5, 6, 7, 8],
  middle: [0, 9, 10, 11, 12],
  ring: [0, 13, 14, 15, 16],
  pinky: [0, 17, 18, 19, 20],
};

/** The palm outline, which no finger chain covers. */
const PALM = [0, 5, 9, 13, 17, 0];

const COLORS = {
  bone: "#7f93ad",
  joint: "#dce6f2",
  flagged: "#f5a524",
  correct: "#3ecf8e",
};

export function resizeOverlay(canvas: HTMLCanvasElement, width: number, height: number): void {
  // Match the canvas backing store to the frame so one unit is one pixel.
  canvas.width = width;
  canvas.height = height;
}

export function clearOverlay(canvas: HTMLCanvasElement): void {
  canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
}

export function drawHand(
  canvas: HTMLCanvasElement,
  hand: DetectedHand,
  options: { hints: Hint[]; correct: boolean },
): void {
  const context = canvas.getContext("2d");
  if (!context) return;

  const { width, height } = canvas;
  context.clearRect(0, 0, width, height);

  const flagged = new Set(options.hints.map((hint) => hint.finger));
  const scale = (index: number): [number, number] => {
    const point = hand.landmarks[index] ?? [0, 0, 0];
    return [point[0] * width, point[1] * height];
  };

  context.lineCap = "round";
  stroke(context, PALM.map(scale), COLORS.bone, Math.max(2, width / 320));

  for (const [finger, chain] of Object.entries(FINGER_CHAINS)) {
    const color = flagged.has(finger)
      ? COLORS.flagged
      : options.correct
        ? COLORS.correct
        : COLORS.bone;
    // A flagged finger is drawn thicker as well as coloured, so the hint is
    // legible to someone who cannot distinguish the two colours.
    const weight = flagged.has(finger) ? Math.max(5, width / 160) : Math.max(3, width / 260);
    stroke(context, chain.map(scale), color, weight);
  }

  context.fillStyle = COLORS.joint;
  for (let index = 0; index < hand.landmarks.length; index += 1) {
    const [x, y] = scale(index);
    context.beginPath();
    context.arc(x, y, Math.max(2, width / 340), 0, Math.PI * 2);
    context.fill();
  }
}

function stroke(
  context: CanvasRenderingContext2D,
  points: [number, number][],
  color: string,
  weight: number,
): void {
  const [first, ...rest] = points;
  if (!first) return;

  context.strokeStyle = color;
  context.lineWidth = weight;
  context.beginPath();
  context.moveTo(first[0], first[1]);
  for (const [x, y] of rest) context.lineTo(x, y);
  context.stroke();
}
