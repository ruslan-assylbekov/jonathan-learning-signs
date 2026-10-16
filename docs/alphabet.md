# Alphabet (RSL dactyl)

33 signs, one per letter of the Russian alphabet, plus a "no sign" class.

## Source

Bukva (Kvanchiani et al., 2024, arXiv:2410.08675). The paper's text does not
name the dynamic letters; it states only:

> Based on expert consultation, 8 letters are dynamically displayed, but due to
> uncertainty in the RSL alphabet, the authors of [17] consider only 7 letters
> are shown with movement.

and, in the caption of Figure 1:

> Each of the RSL alphabet's 33 signs corresponds to a letter of the Russian
> language alphabet. Arrows indicate the direction of movement for 8 dynamic
> signs.

The list below was therefore **read off Figure 1** — the eight signs drawn with
a motion arrow. The count matches the caption's eight. Nothing here is from
memory; no handshape is described, because the MVP does not need descriptions.

**Caveat.** This is a reading of a line drawing, and the paper itself records
that another source counts seven. Cross-check it empirically once landmarks are
extracted: per-class landmark displacement across a clip separates moving signs
from still ones by measurement rather than by eye. If that disagrees with this
table, the measurement wins and this file is corrected.

## Dynamic — excluded from the MVP (8)

Д, Ё, З, Й, Ц, Щ, Ъ, Ь

A single frame cannot distinguish these, so a static-frame classifier cannot
honestly recognize them (CLAUDE.md 2, Tier 3 item 13).

## Static — the MVP subset (25)

А, Б, В, Г, Е, Ж, И, К, Л, М, Н, О, П, Р, С, Т, У, Ф, Х, Ч, Ш, Ы, Э, Ю, Я

Note from the paper, worth remembering when reading the confusion matrix:

> "Б" – "Ф" and "В" – "Ш" have comparable configurations and are distinguished
> only by the spacing between fingers.

## The no-sign class

Bukva labels it `no_event` (105 clips). The code calls it `no_sign`
(`stabilizer.NO_SIGN`), so extraction maps one to the other in `labels.json`.
Those 105 clips exist only under `trimmed/`, which is why the annotation table
has 3,862 rows while `original/` holds the 3,757 the paper counts.
