/**
 * Client for /ws/practice.
 *
 * Mirrors the protocol in CLAUDE.md 7 exactly; any change to it belongs in the
 * same commit as the backend schemas and the integration tests.
 */

import type { DetectedHand } from "./landmarker";

/** Where a finger should end up (backend feedback.py, CLAUDE.md 6.5). */
export type HintCode =
  | "STRAIGHTEN"
  | "BEND_HALFWAY"
  | "ONLY_HALFWAY"
  | "BEND_FULLY"
  | "BEND_MORE"
  | "BEND_LESS";

export interface Hint {
  readonly finger: string;
  readonly code: HintCode;
}

export interface EvalMessage {
  readonly type: "eval";
  readonly seq: number;
  readonly pred: string;
  readonly confidence: number;
  readonly correct: boolean;
  readonly hold_progress: number;
  readonly hints: Hint[];
  readonly server_ms: number;
  readonly t_client: number;
}

export interface SuccessMessage {
  readonly type: "success";
  readonly letter_id: number;
  readonly time_to_success_ms: number;
}

export interface ErrorMessage {
  readonly type: "error";
  readonly code: string;
  readonly detail: string;
}

type ServerMessage = EvalMessage | SuccessMessage | ErrorMessage;

export interface PracticeHandlers {
  onEval(message: EvalMessage, roundTripMs: number): void;
  onSuccess(message: SuccessMessage): void;
  onError(message: ErrorMessage): void;
  onClose(): void;
}

/**
 * The client throttles to this rate (CLAUDE.md 7). The camera runs at 30 Hz or
 * more; evaluating every frame would spend bandwidth and server CPU on frames
 * a learner cannot react to anyway. The server drops anything above its own
 * ceiling regardless.
 */
const SEND_HZ = 15;
const MIN_SEND_INTERVAL_MS = 1000 / SEND_HZ;

/**
 * Coordinates are rounded to 4 decimals before sending: 1/10,000 of the frame,
 * 0.13 px at 1280, far below MediaPipe's own jitter. It cuts a frame message
 * from 1,395 to 592 bytes; on the held-out test signers accuracy moved from
 * 0.7020 to 0.7017 with no frame made invalid. 2 decimals is too coarse --
 * joints collapse onto each other. See experiments/bandwidth.
 */
const COORDINATE_DECIMALS = 4;
const SCALE = 10 ** COORDINATE_DECIMALS;

function round(value: number): number {
  return Math.round(value * SCALE) / SCALE;
}

export class PracticeSocket {
  private socket: WebSocket;
  private seq = 0;
  private lastSentAt = 0;
  private started = false;

  private constructor(socket: WebSocket, handlers: PracticeHandlers) {
    this.socket = socket;
    socket.addEventListener("message", (event) => {
      const message = JSON.parse(event.data as string) as ServerMessage;
      switch (message.type) {
        case "eval":
          handlers.onEval(message, Date.now() - message.t_client);
          break;
        case "success":
          handlers.onSuccess(message);
          break;
        case "error":
          handlers.onError(message);
          break;
      }
    });
    socket.addEventListener("close", () => handlers.onClose());
  }

  static connect(token: string, handlers: PracticeHandlers): Promise<PracticeSocket> {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(`${protocol}//${location.host}/ws/practice`);

    return new Promise((resolve, reject) => {
      socket.addEventListener("open", () => {
        // Authenticate in the first message rather than a query parameter, so
        // the token never reaches an access log (CLAUDE.md 7).
        socket.send(JSON.stringify({ type: "auth", token }));
        resolve(new PracticeSocket(socket, handlers));
      });
      socket.addEventListener("error", () =>
        reject(new Error("Could not reach the server's practice socket.")),
      );
    });
  }

  /**
   * Begin a target. Frame size goes here, not in every frame, so the per-frame
   * payload stays minimal; re-send it whenever the resolution changes
   * (CLAUDE.md 7).
   */
  start(letterId: number, width: number, height: number): void {
    this.send({ type: "start", mode: "learn", letter_id: letterId, width, height });
    this.started = true;
    this.lastSentAt = 0;
  }

  stop(): void {
    if (this.started) this.send({ type: "stop" });
    this.started = false;
  }

  /** Send one frame unless it falls inside the throttle window. */
  sendFrame(hand: DetectedHand, now = Date.now()): boolean {
    if (!this.started || now - this.lastSentAt < MIN_SEND_INTERVAL_MS) return false;
    this.lastSentAt = now;
    this.send({
      type: "frame",
      seq: this.seq++,
      t_client: now,
      handedness: hand.handedness,
      mirrored: false, // the pixels are unflipped; see landmarker.ts
      landmarks: hand.landmarks.map(([x, y, z]) => [round(x), round(y), round(z)]),
    });
    return true;
  }

  close(): void {
    this.stop();
    this.socket.close();
  }

  private send(payload: unknown): void {
    if (this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(payload));
    }
  }
}
