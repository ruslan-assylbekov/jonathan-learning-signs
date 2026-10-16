"""Explore the Bukva dataset before writing any processing code.

CLAUDE.md 8.1: do not assume the annotation format -- read it from the data.
So this script hardcodes as little as possible. It discovers the annotation
file, profiles whatever columns are actually there, and only then probes for
the fields the pipeline will need, reporting what is missing instead of
crashing.

What it answers:

* What is in the archive -- directories, file types, counts, size.
* What the annotation table looks like -- columns, dtypes, nulls, cardinality.
* Do annotation rows and video files on disk agree?
* How are classes, signers and the official split distributed, and do signers
  leak between train and test (which would invalidate CLAUDE.md 8.4)?
* What frame geometry to expect -- resolutions and aspect ratios, which
  feature extraction now depends on (CLAUDE.md 6.2, step 4).
* Is a usable sign interval annotated, and how long are the clips?

Usage:
    uv run --group ml python ml/bukva/inspect_dataset.py --bukva-dir PATH

The path is passed in, never inferred from the working directory: the dataset
is downloaded data and stays outside the repo.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
TABLE_SUFFIXES = {".tsv", ".csv", ".txt", ".json", ".jsonl", ".parquet"}

#: Fields the pipeline will need, and the column names worth looking for. Each
#: is reported as found or missing -- never assumed.
WANTED = {
    "video id": ("attachment_id", "video_id", "id", "filename", "file", "name"),
    "signer id": ("user_id", "signer_id", "participant", "subject", "user"),
    "class": ("text", "label", "letter", "class", "gloss", "sign"),
    "interval start": ("begin", "start", "start_frame", "begin_frame"),
    "interval end": ("end", "end_frame", "finish"),
    "split": ("train", "split", "subset", "is_train"),
    "width": ("width", "w", "frame_width"),
    "height": ("height", "h", "frame_height"),
    "length": ("length", "duration", "n_frames", "frames"),
}

MAX_LISTED_VALUES = 40
LOW_CARDINALITY = 60


def main() -> None:
    use_utf8_output()
    args = parse_args()
    root = args.bukva_dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}\nSet BUKVA_DIR or pass --bukva-dir.")

    report: dict[str, Any] = {"root": str(root)}
    section("Archive layout")
    report["layout"] = describe_layout(root)

    tables = find_tables(root)
    if not tables:
        raise SystemExit("no annotation table found (.tsv/.csv/.json/.parquet) under the root")

    section("Candidate annotation files")
    for path in tables:
        print(f"  {path.relative_to(root)}  ({path.stat().st_size / 1e6:.2f} MB)")

    annotations = args.annotations or tables[0]
    shown = annotations.relative_to(root) if annotations.is_relative_to(root) else annotations
    section(f"Reading {shown}")
    frame = load_table(annotations)
    print(f"  {len(frame)} rows x {len(frame.columns)} columns")

    report["annotations"] = {"path": str(annotations), "rows": int(len(frame))}
    report["columns"] = profile_columns(frame)
    report["wanted"] = resolve_wanted(frame)
    report["videos"] = cross_check_videos(root, frame, report["wanted"].get("video id"))
    report["distributions"] = describe_distributions(frame, report["wanted"])
    report["geometry"] = describe_geometry(frame, report["wanted"])
    report["intervals"] = describe_intervals(frame, report["wanted"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    section("Next steps")
    print(f"  Report written to {args.out}")
    print("  Decide the static letter subset from the reference, then fill docs/alphabet.md.")
    print("  Staticness cannot be read from these annotations -- do not guess it here.")


def use_utf8_output() -> None:
    """Print Cyrillic class names instead of question marks.

    A Windows console defaults to a legacy code page, which mangles every
    letter of the alphabet this project is about.
    """
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bukva-dir",
        type=Path,
        default=Path(os.environ.get("BUKVA_DIR", "")),
        help="unpacked dataset root (default: $BUKVA_DIR)",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help="annotation file to read (default: the largest table found)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("ml/data/inspection_report.json"),
        help="where to write the machine-readable report",
    )
    return parser.parse_args()


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def describe_layout(root: Path) -> dict[str, Any]:
    suffixes: Counter[str] = Counter()
    total_bytes = 0
    per_dir: Counter[str] = Counter()

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffixes[path.suffix.lower()] += 1
        total_bytes += path.stat().st_size
        top = path.relative_to(root).parts[0] if len(path.relative_to(root).parts) > 1 else "."
        per_dir[top] += 1

    print(f"  {sum(suffixes.values())} files, {total_bytes / 1e9:.2f} GB total")
    print("  by top-level entry:")
    for name, count in per_dir.most_common():
        print(f"    {name:<30} {count:>7} files")
    print("  by extension:")
    for suffix, count in suffixes.most_common(15):
        print(f"    {suffix or '(none)':<30} {count:>7}")

    return {
        "files": sum(suffixes.values()),
        "bytes": total_bytes,
        "by_top_level": dict(per_dir),
        "by_suffix": dict(suffixes),
    }


def find_tables(root: Path) -> list[Path]:
    tables = [
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in TABLE_SUFFIXES
    ]
    return sorted(tables, key=lambda path: path.stat().st_size, reverse=True)


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")

    # Let pandas sniff the delimiter rather than trusting the extension: a file
    # named .csv holding tabs is a classic way to lose an afternoon.
    return pd.read_csv(path, sep=None, engine="python")


def profile_columns(frame: pd.DataFrame) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    section("Columns")

    for column in frame.columns:
        values = frame[column]
        unique = int(values.nunique(dropna=True))
        nulls = int(values.isna().sum())
        print(f"  {column!r}: dtype={values.dtype} unique={unique} nulls={nulls}")

        entry: dict[str, Any] = {"dtype": str(values.dtype), "unique": unique, "nulls": nulls}
        if unique <= LOW_CARDINALITY:
            counts = values.value_counts(dropna=False)
            entry["value_counts"] = {str(key): int(count) for key, count in counts.items()}
            print(f"      values: {format_values(counts.index[:MAX_LISTED_VALUES])}")
        else:
            entry["examples"] = [str(value) for value in values.dropna().head(5)]
            print(f"      examples: {entry['examples']}")
        profile[str(column)] = entry

    section("First rows")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(frame.head(5).to_string())

    return profile


def format_values(values: Iterable[Any]) -> str:
    return ", ".join(repr(value) for value in values)


def resolve_wanted(frame: pd.DataFrame) -> dict[str, str | None]:
    """Map each field the pipeline needs to an actual column, or to None."""
    lowered = {str(column).lower(): str(column) for column in frame.columns}
    found: dict[str, str | None] = {}

    section("Fields the pipeline needs")
    for field, candidates in WANTED.items():
        match = next((lowered[name] for name in candidates if name in lowered), None)
        found[field] = match
        status = f"-> {match!r}" if match else "MISSING"
        print(f"  {field:<16} {status}")

    if any(column is None for column in found.values()):
        print("\n  Missing fields are not an error yet -- check the column list above")
        print("  for a differently named equivalent before changing the pipeline.")
    return found


def cross_check_videos(root: Path, frame: pd.DataFrame, id_column: str | None) -> dict[str, Any]:
    section("Videos on disk vs annotation rows")
    on_disk = {
        path.name: path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    }
    print(f"  {len(on_disk)} video files found")

    stems = {Path(name).stem: name for name in on_disk}
    result: dict[str, Any] = {"files_on_disk": len(on_disk)}

    if id_column is None:
        print("  no id column resolved, skipping the match")
        return result

    ids = frame[id_column].dropna().astype(str)
    matched = sum(1 for value in ids if value in on_disk or Path(value).stem in stems)
    missing = [value for value in ids if value not in on_disk and Path(value).stem not in stems]

    print(f"  {matched}/{len(ids)} annotation rows matched a file")
    if missing:
        print(f"  first unmatched ids: {missing[:5]}")
        print("  (ids may need a suffix or a subdirectory prefix -- check the paths above)")

    result |= {"rows": int(len(ids)), "matched": int(matched), "unmatched": len(missing)}
    return result


def describe_distributions(frame: pd.DataFrame, wanted: dict[str, str | None]) -> dict[str, Any]:
    result: dict[str, Any] = {}

    class_column = wanted.get("class")
    if class_column:
        section(f"Classes ({class_column})")
        counts = frame[class_column].value_counts()
        print(f"  {len(counts)} classes, {counts.min()}-{counts.max()} samples each")
        for value, count in counts.items():
            print(f"    {str(value):<12} {count:>5}")
        result["classes"] = {str(key): int(count) for key, count in counts.items()}

    signer_column = wanted.get("signer id")
    if signer_column:
        section(f"Signers ({signer_column})")
        per_signer = frame[signer_column].value_counts()
        print(f"  {len(per_signer)} signers, {per_signer.min()}-{per_signer.max()} videos each")
        result["signers"] = {"count": int(len(per_signer))}

    split_column = wanted.get("split")
    if split_column:
        section(f"Official split ({split_column})")
        counts = frame[split_column].value_counts(dropna=False)
        for value, count in counts.items():
            print(f"    {str(value):<12} {count:>6} rows")
        result["split"] = {str(key): int(count) for key, count in counts.items()}

        if signer_column:
            groups = frame.groupby(split_column)[signer_column].apply(set)
            leaked = set.intersection(*groups) if len(groups) > 1 else set()
            print(f"  signers per split: {[len(group) for group in groups]}")
            print(f"  signers in more than one split: {len(leaked)}")
            if leaked:
                print("  WARNING: the official split is not signer-disjoint. CLAUDE.md 8.4")
                print("  requires signer-disjoint evaluation -- build our own groups.")
            result["split_signer_overlap"] = len(leaked)

    return result


def describe_geometry(frame: pd.DataFrame, wanted: dict[str, str | None]) -> dict[str, Any]:
    width_column, height_column = wanted.get("width"), wanted.get("height")
    if not (width_column and height_column):
        return {}

    section("Frame geometry")
    sizes = (
        frame[[width_column, height_column]]
        .dropna()
        .astype(int)
        .value_counts()
        .reset_index(name="videos")
    )
    for row in sizes.itertuples(index=False):
        width, height, videos = row[0], row[1], row[2]
        print(f"    {width:>5} x {height:<5} aspect {width / height:.4f}   {videos:>6} videos")

    aspects = (frame[width_column] / frame[height_column]).dropna()
    print(f"  distinct aspect ratios: {aspects.nunique()}")
    print("  feature extraction corrects for this (CLAUDE.md 6.2, step 4), so every")
    print("  extracted row must carry the width and height of its own video.")

    return {
        "resolutions": [
            {"width": int(row[0]), "height": int(row[1]), "videos": int(row[2])}
            for row in sizes.itertuples(index=False)
        ],
        "distinct_aspects": int(aspects.nunique()),
    }


def describe_intervals(frame: pd.DataFrame, wanted: dict[str, str | None]) -> dict[str, Any]:
    begin_column, end_column = wanted.get("interval start"), wanted.get("interval end")
    length_column = wanted.get("length")
    result: dict[str, Any] = {}

    if length_column:
        section(f"Clip length ({length_column})")
        print(frame[length_column].describe().to_string())
        result["length"] = {
            key: float(value) for key, value in frame[length_column].describe().items()
        }

    if not (begin_column and end_column):
        print("\n  No sign interval columns resolved. If the trimmed videos are already")
        print("  cut to the sign, extraction can use whole clips -- confirm from the")
        print("  layout above, and document the choice (CLAUDE.md 8.2).")
        return result

    section(f"Sign interval ({begin_column}, {end_column})")
    usable = frame[[begin_column, end_column]].dropna()
    print(f"  {len(usable)}/{len(frame)} rows have both bounds")

    if usable.empty:
        return result

    span = usable[end_column] - usable[begin_column]
    print(f"  interval span: min={span.min()} median={span.median()} max={span.max()}")
    invalid = int((span <= 0).sum())
    print(f"  non-positive spans: {invalid}")
    if invalid:
        print("  WARNING: these rows cannot be sampled from; log and skip them.")

    result["interval"] = {
        "rows_with_bounds": int(len(usable)),
        "span_min": float(span.min()),
        "span_median": float(span.median()),
        "span_max": float(span.max()),
        "non_positive": invalid,
    }
    return result


if __name__ == "__main__":
    main()
