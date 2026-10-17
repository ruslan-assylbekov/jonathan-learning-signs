/**
 * Learn mode: show a target, evaluate the camera, correct the learner.
 *
 * Holds the render loop and the DOM; every decision about whether a sign is
 * right is the backend's (CLAUDE.md 6.1).
 */

import { startCamera, type CameraStream } from "../camera";
import { demoExtendedFingers, errorText, hintText, isDemoLabel, targetName } from "../i18n";
import { createLandmarker, detectHand, type DetectedHand } from "../landmarker";
import { clearOverlay, drawHand, resizeOverlay } from "../overlay";
import { PracticeSocket, type EvalMessage, type Hint } from "../ws-client";
import type { HandLandmarker } from "@mediapipe/tasks-vision";

interface Letter {
  id: number;
  symbol: string;
  sort_order: number;
}

export class LearnPage {
  private readonly video = element<HTMLVideoElement>("video");
  private readonly overlay = element<HTMLCanvasElement>("overlay");
  private readonly startButton = element<HTMLButtonElement>("start");
  private readonly targetSelect = element<HTMLSelectElement>("target");
  private readonly status = element<HTMLElement>("status");
  private readonly feedback = element<HTMLElement>("feedback");
  private readonly hintList = element<HTMLElement>("hints");
  private readonly holdBar = element<HTMLElement>("hold-bar");
  private readonly detected = element<HTMLElement>("detected");
  private readonly stats = element<HTMLElement>("stats");
  private readonly banner = element<HTMLElement>("banner");

  private letters: Letter[] = [];
  private camera: CameraStream | null = null;
  private landmarker: HandLandmarker | null = null;
  private socket: PracticeSocket | null = null;
  private running = false;
  private hints: Hint[] = [];
  private correct = false;
  private lastFrameAt = 0;
  private fps = 0;
  /** Frame size the server was last told about in a start message. */
  private frameWidth = 0;
  private frameHeight = 0;
  /** The delayed move to the next letter after a success, if one is pending. */
  private pendingAdvance: number | null = null;

  /**
   * @param token JWT from signing in; sent as the socket's first message.
   * @param onSessionEnded called when the server rejects the token -- it
   *   expired, or the secret changed -- so the page can ask to sign in again.
   */
  constructor(
    private readonly token: string,
    private readonly onSessionEnded: () => void,
  ) {}

  async init(): Promise<void> {
    this.startButton.addEventListener("click", () => void this.toggle());
    this.targetSelect.addEventListener("change", () => {
      // A manual choice wins over an automatic advance still waiting to fire.
      this.cancelPendingAdvance();
      this.retarget();
    });

    try {
      this.letters = await fetchLetters();
    } catch (error) {
      this.say(this.status, `Backend unreachable: ${message(error)}`);
      return;
    }

    this.fillTargets();
    this.startButton.disabled = false;
    this.say(this.status, "Ready. Start the camera when you are.");
  }

  private fillTargets(): void {
    this.targetSelect.replaceChildren(
      ...this.letters.map((letter) => {
        const option = document.createElement("option");
        option.value = String(letter.id);
        option.textContent = targetName(letter.symbol);
        return option;
      }),
    );

    // Say plainly that these are not sign language yet, so a screenshot of this
    // screen can never be mistaken for a working RSL recogniser.
    const stub = this.letters.some((letter) => isDemoLabel(letter.symbol));
    this.banner.hidden = !stub;
    this.drawTargetHint();
  }

  private get target(): Letter | undefined {
    const id = Number(this.targetSelect.value);
    return this.letters.find((letter) => letter.id === id);
  }

  private drawTargetHint(): void {
    const symbol = this.target?.symbol ?? "";
    const extended = demoExtendedFingers(symbol);
    const names = ["thumb", "index", "middle", "ring", "little"];
    element<HTMLElement>("target-detail").textContent = extended
      ? `extend: ${names.filter((_, index) => extended[index]).join(", ") || "nothing"}`
      : symbol;
  }

  private async toggle(): Promise<void> {
    if (this.running) {
      this.teardown();
      return;
    }
    await this.launch();
  }

  private async launch(): Promise<void> {
    this.startButton.disabled = true;
    try {
      if (!this.landmarker) {
        // Created once and reused: each detector holds WASM and GPU memory,
        // and a new one per Start would leak it on every restart.
        this.say(this.status, "Loading the hand model…");
        this.landmarker = await createLandmarker();
      }

      this.say(this.status, "Asking for the camera…");
      this.camera = await startCamera(this.video);
      this.frameWidth = this.camera.width;
      this.frameHeight = this.camera.height;
      resizeOverlay(this.overlay, this.frameWidth, this.frameHeight);

      this.say(this.status, "Connecting…");
      this.socket = await PracticeSocket.connect(this.token, {
        onEval: (message, roundTrip) => this.onEval(message, roundTrip),
        onSuccess: () => this.onSuccess(),
        onError: (error) => {
          if (error.code === "UNAUTHENTICATED") {
            this.teardown();
            this.onSessionEnded();
            return;
          }
          this.say(this.feedback, errorText(error.code, error.detail), "fix");
        },
        onClose: () => {
          // Closing it ourselves (Stop) has already torn down; anything else
          // means the server went away while the camera was still running.
          if (!this.running) return;
          this.teardown();
          this.say(this.status, "Connection lost. Press Start camera to reconnect.", "fix");
        },
      });
      this.retarget();

      this.running = true;
      this.startButton.textContent = "Stop";
      this.startButton.disabled = false;
      this.say(this.status, `Live · ${this.frameWidth}×${this.frameHeight}`);
      requestAnimationFrame((time) => this.loop(time));
    } catch (error) {
      this.say(this.status, message(error), "fix");
      this.teardown();
      this.startButton.disabled = false;
    }
  }

  private retarget(): void {
    this.drawTargetHint();
    const target = this.target;
    if (!this.socket || !this.camera || !target) return;

    this.socket.start(target.id, this.frameWidth, this.frameHeight);
    this.hints = [];
    this.correct = false;
    this.holdBar.style.width = "0%";
    this.say(this.feedback, `Show: ${targetName(target.symbol)}`);
  }

  private loop(timestamp: number): void {
    if (!this.running || !this.camera || !this.landmarker) return;

    this.followResolution();

    const hand = detectHand(this.landmarker, this.camera.video, timestamp);
    if (hand) {
      this.onHand(hand, timestamp);
    } else {
      // No frame is sent, so the server's hold count stands; the bar is left
      // where it is rather than pretending the progress was lost.
      clearOverlay(this.overlay);
      this.detected.textContent = "no hand in frame";
    }

    requestAnimationFrame((next) => this.loop(next));
  }

  /**
   * Re-send start when the camera's frame size changes -- a phone rotating,
   * another camera picked. Landmarks are normalized against the frame, so
   * frames sent under a stale size would be scaled wrongly (CLAUDE.md 7).
   */
  private followResolution(): void {
    const { videoWidth, videoHeight } = this.video;
    if (!videoWidth || !videoHeight) return;
    if (videoWidth === this.frameWidth && videoHeight === this.frameHeight) return;

    this.frameWidth = videoWidth;
    this.frameHeight = videoHeight;
    resizeOverlay(this.overlay, videoWidth, videoHeight);
    this.say(this.status, `Live · ${videoWidth}×${videoHeight}`);
    this.retarget();
  }

  private onHand(hand: DetectedHand, timestamp: number): void {
    drawHand(this.overlay, hand, { hints: this.hints, correct: this.correct });
    this.socket?.sendFrame(hand);

    if (this.lastFrameAt) {
      // Smooth the camera rate so the number is readable rather than jittering.
      this.fps = this.fps * 0.9 + (1000 / (timestamp - this.lastFrameAt)) * 0.1;
    }
    this.lastFrameAt = timestamp;
  }

  private onEval(message: EvalMessage, roundTripMs: number): void {
    this.hints = message.hints;
    this.correct = message.correct;

    this.holdBar.style.width = `${Math.round(message.hold_progress * 100)}%`;
    this.detected.textContent = `${targetName(message.pred)} · ${(message.confidence * 100).toFixed(0)}%`;
    this.stats.textContent = `camera ${this.fps.toFixed(0)} fps · server ${message.server_ms.toFixed(1)} ms · round trip ${roundTripMs} ms`;

    if (message.correct) {
      this.say(this.feedback, "Hold it…", "good");
      this.hintList.replaceChildren();
      return;
    }

    this.say(this.feedback, `Show: ${targetName(this.target?.symbol ?? "")}`);
    this.hintList.replaceChildren(
      ...message.hints.map((hint) => {
        const item = document.createElement("li");
        item.textContent = hintText(hint.finger, hint.code);
        return item;
      }),
    );
  }

  private onSuccess(): void {
    this.say(this.feedback, "Correct.", "good");
    this.hintList.replaceChildren();

    // Let the learner see the result before the target changes. The selection
    // moves only when the server is told, so the two never disagree.
    this.cancelPendingAdvance();
    this.pendingAdvance = window.setTimeout(() => {
      this.pendingAdvance = null;
      const index = this.letters.findIndex((letter) => letter.id === this.target?.id);
      const next = this.letters[(index + 1) % this.letters.length];
      if (!next) return;
      this.targetSelect.value = String(next.id);
      this.retarget();
    }, 900);
  }

  private cancelPendingAdvance(): void {
    if (this.pendingAdvance !== null) window.clearTimeout(this.pendingAdvance);
    this.pendingAdvance = null;
  }

  private teardown(): void {
    this.running = false;
    this.cancelPendingAdvance();
    this.socket?.close();
    this.camera?.stop();
    this.socket = null;
    this.camera = null;
    clearOverlay(this.overlay);
    this.startButton.textContent = "Start camera";
    this.say(this.status, "Stopped.");
  }

  private say(target: HTMLElement, text: string, tone: "good" | "fix" | "" = ""): void {
    target.textContent = text;
    target.className = tone;
  }
}

async function fetchLetters(): Promise<Letter[]> {
  const response = await fetch("/letters");
  if (!response.ok) throw new Error(`/letters returned ${response.status}`);
  return (await response.json()) as Letter[];
}

function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`markup is missing #${id}`);
  return found as T;
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
