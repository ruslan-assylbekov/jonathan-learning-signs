# Experiment: bandwidth (CLAUDE.md 16.1)

## Question

How much network traffic does sending hand landmarks cost, compared with the
obvious alternative of uploading camera frames for server-side recognition?
This is the quantitative case for running MediaPipe in the browser.

## Hypothesis

A landmark message is a few hundred floats; a JPEG frame is tens of kilobytes.
Landmarks should be one to two orders of magnitude smaller, for the same
recognition, while also keeping video on the user's device.

## Setup

- **Landmark messages** are rebuilt byte for byte as `frontend/src/ws-client.ts`
  sends them (`JSON.stringify`, same key order), from 1,000 real frames per
  seed in `ml/data/bukva_landmarks.parquet`. Numbers use ECMAScript's
  Number-to-string layout; `run.py`'s `js_number` was checked against Node's
  `String(x)` on 15,616 values with no mismatch. Since 2026-10-19 the client
  rounds coordinates to 4 decimals (`Math.round(x * 1e4) / 1e4`); `js_round4`
  reproduces it and matched Node on 23,905 values. Full-precision sizes are
  kept in a separate column for comparison.
- **Server replies** come from the real pipeline: model-v2, the shipped hint
  templates and the production `EvalOut` serializer, with the learner
  attempting the letter they are showing.
- **JPEG frames** are the middle frame of 60 random Bukva clips per seed,
  resized to a long side of 640 and 1280 px (typical webcam pixel counts) and
  encoded by OpenCV at quality 50, 75 and 90. Nothing is sent to the app
  (CLAUDE.md 7).
- **Repetitions:** three independent random samples (seeds 0, 1, 2). Encoding
  is deterministic, so re-running one sample would only repeat its numbers;
  separate samples show whether the result depends on which frames were drawn.
- **Wire size** adds RFC 6455 WebSocket framing (client frames are masked).
  TLS record overhead is not included.
- **Precision check** (`precision.py`): accuracy of model-v2 on the held-out
  test signers when landmarks are rounded before sending. This is what chose
  4 decimals: the accuracy change is in the fourth digit, while 2 decimals
  makes frames invalid.

## How to run

```bash
BUKVA_DIR=/path/to/bukva uv run --project backend --group ml python experiments/bandwidth/run.py
uv run --project backend --group ml python experiments/bandwidth/precision.py
uv run --project backend --group ml python experiments/bandwidth/analyze.py
```

## Results

Generated, never hand-edited: [`results/summary.md`](results/summary.md), from
`results/landmarks.csv`, `results/jpeg.csv`, `results/precision.csv` and
`results/environment.json`, with the plot `results/bandwidth.png`.

## Limitations

- JPEG sizes come from OpenCV's libjpeg. A browser would encode with
  `canvas.toBlob`, a different encoder at the same nominal quality.
- Bukva clips are phone video of people indoors; a webcam scene may compress
  differently. The three seeds bracket the variation across Bukva's scenes.
- The JPEG comparison is upload only, at the same 15 Hz the client uses, and
  at 30 fps for streaming every frame. A server-side design would also need
  replies, which are small in both designs.
