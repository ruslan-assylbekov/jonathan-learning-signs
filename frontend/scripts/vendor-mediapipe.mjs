/**
 * Put MediaPipe's runtime into public/mediapipe/.
 *
 * CLAUDE.md 14, pitfall 6: self-host the wasm and the .task model so a demo
 * never depends on a CDN being reachable. The wasm ships inside the npm
 * package, so it is copied; the model is not, so it is downloaded once.
 *
 * Both are third-party binaries and stay out of git (see .gitignore); run
 * `npm run vendor` after a fresh clone.
 */

import { createHash } from "node:crypto";
import { cp, mkdir, stat, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const frontend = join(here, "..");
const target = join(frontend, "public", "mediapipe");

const WASM_SOURCE = join(frontend, "node_modules", "@mediapipe", "tasks-vision", "wasm");
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const MODEL_FILE = join(target, "hand_landmarker.task");

async function exists(path) {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

async function vendorWasm() {
  if (!(await exists(WASM_SOURCE))) {
    throw new Error(`${WASM_SOURCE} is missing — run npm install first.`);
  }
  await cp(WASM_SOURCE, join(target, "wasm"), { recursive: true });
  console.log("wasm runtime copied from node_modules");
}

async function vendorModel() {
  if (await exists(MODEL_FILE)) {
    console.log("hand_landmarker.task already present, leaving it alone");
    return;
  }

  console.log(`downloading ${MODEL_URL}`);
  const response = await fetch(MODEL_URL);
  if (!response.ok) {
    throw new Error(`download failed: ${response.status} ${response.statusText}`);
  }

  const bytes = Buffer.from(await response.arrayBuffer());
  await writeFile(MODEL_FILE, bytes);
  const digest = createHash("sha256").update(bytes).digest("hex");
  console.log(`hand_landmarker.task  ${(bytes.length / 1e6).toFixed(2)} MB  sha256 ${digest}`);
}

await mkdir(target, { recursive: true });
await vendorWasm();
await vendorModel();
console.log("done");
