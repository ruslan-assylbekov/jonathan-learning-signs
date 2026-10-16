/**
 * Display text. The backend sends codes, this file turns them into words
 * (CLAUDE.md 6.5), so wording and translation never require a server change.
 *
 * Keys are English because the repository is English (CLAUDE.md 1). Adding a
 * Russian dictionary later means adding one object, not touching call sites.
 */

import type { HintCode } from "./ws-client";

/** Finger names as the backend keys them. */
const FINGERS: Record<string, string> = {
  thumb: "thumb",
  index: "index finger",
  middle: "middle finger",
  ring: "ring finger",
  pinky: "little finger",
};

/**
 * Each code names where the finger should end up, not just a direction:
 * "bend your index finger" is ambiguous when the letter wants it half bent.
 * The thumb only ever gets BEND_MORE / BEND_LESS, because its states cannot be
 * told apart from the angles the backend measures.
 */
const HINTS: Record<HintCode, (finger: string) => string> = {
  STRAIGHTEN: (finger) => `Straighten your ${finger}`,
  BEND_HALFWAY: (finger) => `Bend your ${finger} halfway`,
  ONLY_HALFWAY: (finger) => `Bend your ${finger} only halfway`,
  BEND_FULLY: (finger) => `Bend your ${finger} all the way`,
  BEND_MORE: (finger) => `Bend your ${finger} more`,
  BEND_LESS: (finger) => `Bend your ${finger} less`,
};

export function hintText(finger: string, code: HintCode): string {
  const name = FINGERS[finger] ?? finger;
  const phrase = HINTS[code];
  return phrase ? phrase(name) : `${name}: ${code}`;
}

/**
 * Readable names for the stub classifier's demo classes.
 *
 * These are ordinary gestures, not sign language: the stub refuses to name
 * letters because no model is trained yet (CLAUDE.md 20). A real letter's
 * symbol is shown as it is.
 */
const DEMO_NAMES: Record<string, string> = {
  "demo:11111": "Open hand",
  "demo:00000": "Fist",
  "demo:01000": "Point",
  "demo:01100": "Two fingers",
  "demo:10000": "Thumb up",
  "demo:00001": "Little finger out",
};

export function isDemoLabel(symbol: string): boolean {
  return symbol.startsWith("demo:");
}

export function targetName(symbol: string): string {
  return DEMO_NAMES[symbol] ?? symbol;
}

/** Which fingers a demo label wants extended, for drawing the target. */
export function demoExtendedFingers(symbol: string): boolean[] | null {
  if (!isDemoLabel(symbol)) return null;
  const digits = symbol.slice("demo:".length);
  if (digits.length !== 5 || /[^01]/.test(digits)) return null;
  return [...digits].map((digit) => digit === "1");
}

export const ERRORS: Record<string, string> = {
  UNAUTHENTICATED: "Your session has ended. Please sign in again.",
  NOT_STARTED: "Session was not started.",
  UNKNOWN_LETTER: "The backend does not know that target.",
  INVALID_FRAME: "That frame could not be read.",
  INVALID_MESSAGE: "The backend rejected a message.",
};

export function errorText(code: string, detail: string): string {
  return ERRORS[code] ?? `${code}: ${detail}`;
}
