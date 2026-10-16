# Bukva dataset notes

Source: https://github.com/ai-forever/bukva — Kvanchiani et al., 2024, arXiv:2410.08675.

## License (read from `license/en_us.pdf` in the Bukva repo, 2026-10-16)

"Public license with attribution and conditions reserved". It is **not** a Creative Commons
license, but it is a reworking of CC BY-SA 4.0. Summary (not legal advice; re-read the PDF
before publishing anything):

- Allowed: reproducing, making available, and creating/sharing Adapted Material (§2a).
  No non-commercial restriction is stated. Training a model is therefore allowed.
- Attribution (§3a): keep creator/copyright/license notices, link to the dataset, state that
  changes were made, and link to the license text.
- Share-alike (§3b): Adapted Material (e.g. our landmark Parquet) may be published only under
  the same or a BY-SA-compatible license, with a link to that license.
- Personality, privacy and publicity rights are **not** granted (§2b) — never publish frames or video.

Consequence for this repo: derived landmark files may be committed only with attribution and a
BY-SA-compatible license note. Default remains: do not commit them, regenerate with `make extract`.

## Download

Zip, ~27 GB (original + trimmed HD+ videos + annotations), link in the Bukva README.
Unzip to a folder outside the repo and set `BUKVA_DIR` in `.env`.

## Annotation format — measured, not assumed

Run `make inspect` to regenerate. Findings from the real archive (2026-10-16),
27.08 GB, `BUKVA_DIR=D:/Codes/bukva`:

`annotations.tsv`, 3,862 rows × 9 columns: `attachment_id` (video stem, a UUID),
`user_id` (anonymized signer), `text` (class, Cyrillic), `begin`/`end`, `height`,
`width`, `train` (bool), `length`.

* **3,862 rows, not 3,757.** `trimmed/` holds 3,862 files and matches the
  annotations one-to-one; `original/` holds 3,757. The 105 missing originals are
  exactly the 105 `no_event` rows, which exist only as trimmed clips. The
  paper's "3,757 videos" therefore counts the lettered clips alone.
* **34 classes:** the 33 letters, each with 101–125 samples, plus `no_event`
  (105), which is the paper's "no sign" class. Our code calls it `no_sign`.
* **155 signers**, 4–46 videos each.
* **The official split is signer-disjoint** — 3,182 train rows across 135
  signers, 680 test rows across 20, with no signer in both. It satisfies
  CLAUDE.md 8.4 and can be used as the held-out test set.
* **`begin`/`end` are frame indices into the original clip, and `trimmed/`
  is exactly that span.** Verified on 40 random clips: the trimmed frame count
  equalled `end - begin` every time. So extraction reads trimmed videos whole
  and needs no interval arithmetic. `length` is the original's frame count.
* **Resolutions vary enormously:** 32 distinct sizes, aspect ratios from 0.428
  to 2.087. Most are portrait phone video (1080×1920, 2,339 clips), against a
  typical webcam's 1.778. Annotation `width`/`height` agreed with the actual
  video streams on all 40 clips sampled, but extraction should still read the
  dimensions from the file it opens. Without the aspect correction
  (CLAUDE.md 6.2, step 4) features from this data would not transfer to a
  webcam at all.

## Running the extraction

`make extract`, or the script directly with `--bukva-dir`.

It is bounded by memory, not by cores. Each worker reserves roughly 1 GB of
Windows commit charge for its MediaPipe graph, against a resident set of about
370 MB, so what runs out first is commit (RAM plus page file) — on a 16 GB
machine with a browser and an editor open, two workers fit and four do not.

Exhaustion does not announce itself. OpenCV says "Insufficient memory", but
MediaPipe fails in two other disguises depending on where the allocation
lands: a `RET_CHECK` about `AllocateTensors` when a worker starts, or
`access violation writing 0x20` inside `mp.Image`, which is a null from a
failed allocation being dereferenced. All three are the same problem. Check
free commit before suspecting the data.

The run is resumable: finished chunks are left alone, so an interrupted
attempt only costs the chunks that were in flight.

## Extraction results (2026-10-16)

`ml/data/bukva_landmarks.parquet`, regenerate with `make extract`:

* **27,897 frames from 3,831 of the 3,862 clips**, 155 signers, 34 classes.
  Nothing was unreadable and nothing failed mid-extraction; the 31 missing
  clips simply had no hand detected in any of their sampled frames.
* **16,112 frames were sampled, 1,789 dropped** for no detected hand. Losses
  concentrate where you would expect: `no_sign` (467), then С, Щ, Ш, Ц.
* **The official split survives and stays signer-disjoint** — 23,077 train and
  4,820 test rows, no signer in both.
* **All of it feeds `features.py` cleanly**: 3,000 sampled rows produced finite
  78-dimensional vectors with no rejections, coordinates within [-3.3, 2.7]
  and angles within [0°, 178°].

One thing to weigh before training: **`no_sign` has 373 frames against a median
of 845 per letter**, because a clip of "no sign" often shows no hand at all and
those frames are dropped. The frames that remain are the useful ones — a hand
in view that is not forming a letter — but the class is half the size of the
others and will need weighting, or more frames sampled from those clips.
