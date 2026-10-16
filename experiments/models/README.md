# Experiment: model comparison (CLAUDE.md 16.5)

## Question

Which classifier should turn a 78-dimensional landmark feature vector into one
of 25 static RSL letters or `no_sign`, judged on signers it has never seen, and
within the 5 ms per-frame budget of CLAUDE.md 6.4?

## Setup

- **Data:** `ml/data/bukva_landmarks.parquet` from `make extract` — up to 8
  evenly spaced frames per clip of Bukva's trimmed videos. Dynamic letters
  (Д Ё З Й Ц Щ Ъ Ь, see `docs/alphabet.md`) are excluded; classes are listed in
  `ml/labels.json`.
- **Features:** `app/domain/gesture/features.py`, `FEATURE_VERSION` 2 — the
  same code the server runs, imported, not copied.
- **Selection:** 5-fold `GroupKFold` over signer ids within Bukva's training
  split. No signer is ever on both sides of a fold; `train.py` asserts it.
  Candidates are ranked by mean macro F1, which weighs every letter equally
  rather than rewarding the common ones.
- **Final report:** the chosen model is refit on all training signers and
  scored once on Bukva's official test split, whose signers never appear in
  training. That split takes no part in choosing.
- **Class imbalance:** `no_sign` has roughly half the frames of a letter.
  Logistic regression and the random forest use balanced class weights, the MLP
  balanced sample weights; kNN cannot weight classes and stays as the plain
  baseline.
- **Latency:** `predict_proba` on one frame at a time, 500 frames, single
  thread — how the server calls it. Feature extraction is not included; the
  server's `server_ms` covers the whole path.

## How to run

```bash
make extract     # once, ~1 hour; see ml/bukva/README.md on memory
make train       # selection + final fit -> ml/artifacts/model-vN.*
make eval        # held-out report -> eval-vN.md, confusion-vN.csv
make templates   # per-letter hint templates -> ml/artifacts/templates-v2.json
```

## Results

Generated files only — nothing below is typed in by hand (CLAUDE.md 16):

- `cv_folds.csv` — every candidate, every fold
- `summary.json` — ranking, chosen model, held-out score, latency
- `eval-v1.md` / `eval-v1.json` — per-class F1, confusions, threshold behaviour
- `confusion-v1.csv` — rows true class, columns predicted

## Not yet measured

- Accuracy on own webcam recordings, the domain-shift check of CLAUDE.md 8.4.
  No webcam set has been recorded yet.
- Repeated runs with different seeds. Fold-to-fold spread is in
  `cv_folds.csv`; seed-to-seed spread is not.
