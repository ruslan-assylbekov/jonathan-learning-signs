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

## Annotation format (from the README — verify with `inspect_dataset.py`)

`annotations.tsv`: `attachment_id` (video file), `user_id` (anonymized signer), `text` (class,
Russian), `begin`/`end` (sign interval, original videos only), `height`, `width`, `train`
(bool), `length`. Official split is by signer: 3,097 train / 660 test videos (3,757 total).
