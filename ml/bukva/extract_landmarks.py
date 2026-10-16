"""Turn Bukva videos into a landmark table (CLAUDE.md 8.2).

Reads ``trimmed/`` clips whole: ``make inspect`` established that a trimmed
clip is exactly the annotated ``[begin, end)`` span, so there is no interval
arithmetic here to get wrong. See ``ml/bukva/README.md``.

What is stored is **raw** MediaPipe landmarks, not features. Feature extraction
lives only in ``features.py`` (CLAUDE.md 6.1), and keeping the raw output means
a change to ``FEATURE_VERSION`` costs a retrain, not another pass over 27 GB of
video. ``width``/``height`` travel with every row because normalization needs
the aspect ratio, and this dataset spans 32 different resolutions.

Handedness is stored as the physical hand, with MediaPipe's selfie assumption
undone exactly as the browser does it (CLAUDE.md 7). The rule has to be the
same in both places; applying it in one and not the other would train the model
in a mirrored feature space.

Usage:
    uv run --group ml python ml/bukva/extract_landmarks.py --bukva-dir PATH

Resumable: finished chunks are left alone on a second run.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

NUM_LANDMARKS = 21
DEFAULT_FRAMES_PER_VIDEO = 8
DEFAULT_CHUNK = 60

DEFAULT_WORKERS = 4
"""Deliberately not the CPU count.

Each worker holds its own MediaPipe graph, and TFLite refuses to allocate its
tensors once memory runs short -- which it reports as an unreadable RET_CHECK
failure rather than an out-of-memory error. On a 16 GB machine also running
Docker and a dev server, four workers fit and ten do not. Raise it with
``--workers`` on a roomier machine; the run is resumable, so a failed attempt
costs only the chunks that were in flight.
"""

#: The same .task file the browser loads, so training and inference see the
#: same detector (CLAUDE.md 8.2).
MODEL_RELATIVE = Path("frontend/public/mediapipe/hand_landmarker.task")

#: Bukva's name for the no-sign class; the code calls it no_sign.
BUKVA_NO_SIGN = "no_event"
NO_SIGN = "no_sign"

COORD_COLUMNS = [f"{axis}{index}" for index in range(NUM_LANDMARKS) for axis in "xyz"]


@dataclass
class ChunkReport:
    """Why frames were lost, counted per class rather than in total.

    A class that silently loses most of its frames is a dataset problem, and it
    is invisible in a single overall number.
    """

    videos: int = 0
    frames_kept: Counter[str] = field(default_factory=Counter)
    frames_no_hand: Counter[str] = field(default_factory=Counter)
    frames_low_score: Counter[str] = field(default_factory=Counter)
    videos_unreadable: list[str] = field(default_factory=list)

    def merge(self, other: ChunkReport) -> None:
        self.videos += other.videos
        self.frames_kept += other.frames_kept
        self.frames_no_hand += other.frames_no_hand
        self.frames_low_score += other.frames_low_score
        self.videos_unreadable += other.videos_unreadable


def main() -> None:
    use_utf8_output()
    args = parse_args()

    root = args.bukva_dir.expanduser().resolve()
    annotations = root / "annotations.tsv"
    if not annotations.is_file():
        raise SystemExit(f"no annotations.tsv under {root}. Set BUKVA_DIR or --bukva-dir.")
    if not args.model.is_file():
        raise SystemExit(
            f"hand model not found at {args.model}.\n"
            "Run `npm run vendor` in frontend/ — extraction uses the same .task "
            "file the browser does."
        )

    frame = pd.read_csv(annotations, sep="\t")
    jobs = build_jobs(root, frame, args)
    args.shards.mkdir(parents=True, exist_ok=True)

    done = already_done(args.shards)
    pending = [job for job in jobs if job["video_id"] not in done]
    print(f"{len(jobs)} videos, {len(done)} already extracted, {len(pending)} to do")

    if pending:
        run_chunks(pending, args)

    merge_shards(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bukva-dir", type=Path, default=Path(os.environ.get("BUKVA_DIR", "")))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parents[2] / MODEL_RELATIVE,
        help="hand_landmarker.task (default: the one the frontend serves)",
    )
    parser.add_argument("--out", type=Path, default=Path("ml/data/bukva_landmarks.parquet"))
    parser.add_argument("--shards", type=Path, default=Path("ml/data/shards"))
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=DEFAULT_FRAMES_PER_VIDEO,
        help="evenly spaced frames kept per clip, so one clip cannot dominate",
    )
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, help="process only the first N videos (smoke test)")
    return parser.parse_args()


def use_utf8_output() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def build_jobs(root: Path, frame: pd.DataFrame, args: argparse.Namespace) -> list[dict[str, Any]]:
    jobs = [
        {
            "video_id": str(row.attachment_id),
            "path": str(root / "trimmed" / f"{row.attachment_id}.mp4"),
            "signer_id": str(row.user_id),
            "letter": NO_SIGN if row.text == BUKVA_NO_SIGN else str(row.text),
            "train": bool(row.train),
            "frames_per_video": args.frames_per_video,
            "min_score": args.min_score,
            "model": str(args.model),
        }
        for row in frame.itertuples(index=False)
    ]
    return jobs[: args.limit] if args.limit else jobs


def already_done(shards: Path) -> set[str]:
    """Video ids present in shards written by an earlier run."""
    done: set[str] = set()
    for shard in shards.glob("*.parquet"):
        try:
            done |= set(pd.read_parquet(shard, columns=["video_id"])["video_id"])
        except (OSError, ValueError) as error:
            print(f"  ignoring unreadable shard {shard.name}: {error}")
    return done


def run_chunks(pending: list[dict[str, Any]], args: argparse.Namespace) -> None:
    chunks = [pending[i : i + args.chunk] for i in range(0, len(pending), args.chunk)]
    report = ChunkReport()
    print(f"{len(chunks)} chunks across {args.workers} workers")

    pool = Pool(args.workers, initializer=init_worker, initargs=(str(args.model), args.min_score))
    try:
        for index, (rows, chunk_report) in enumerate(
            pool.imap_unordered(process_chunk, chunks), start=1
        ):
            report.merge(chunk_report)
            if rows:
                shard = args.shards / f"shard-{index:05d}-{os.getpid()}.parquet"
                pd.DataFrame(rows).to_parquet(shard, index=False)
            kept = sum(report.frames_kept.values())
            print(
                f"  chunk {index}/{len(chunks)}: {report.videos} videos, {kept} frames kept",
                flush=True,
            )
    finally:
        # terminate, not close/join: a worker holding MediaPipe's native
        # threads does not always return from a clean shutdown, and the run
        # then hangs after every shard is already safely on disk.
        pool.terminate()
        pool.join()

    write_report(args, report)


#: Built once per worker process by :func:`init_worker`. Creating a detector
#: per chunk instead exhausts TFLite's allocator after a handful of chunks --
#: closing one does not promptly return its memory -- which looks like a
#: parallelism limit but is really a lifetime bug.
_LANDMARKER: Any = None


def init_worker(model: str, min_score: float) -> None:
    global _LANDMARKER
    try:
        _LANDMARKER = make_landmarker(model, min_score)
    except RuntimeError as error:
        # TFLite reports a failed tensor allocation as an opaque RET_CHECK,
        # never as "out of memory", which sends you looking in the wrong place.
        raise RuntimeError(
            "could not start a hand detector in this worker, which usually "
            "means memory ran short. Try fewer --workers (the default is "
            f"{DEFAULT_WORKERS}) or close other heavy processes. Underlying "
            f"error: {error}"
        ) from error


def process_chunk(jobs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], ChunkReport]:
    """One worker's share of the videos, through its process-wide detector."""
    rows: list[dict[str, Any]] = []
    report = ChunkReport()
    for job in jobs:
        extract_one(_LANDMARKER, job, rows, report)
    return rows, report


def make_landmarker(model: str, min_score: float) -> Any:
    """A detector in IMAGE mode.

    IMAGE rather than VIDEO mode, deliberately. VIDEO mode demands
    monotonically increasing timestamps, which clips restarting at zero break,
    and it carries tracking state from the end of one clip into the start of
    the next. IMAGE mode has no such state, so the frames we keep are
    independent estimates -- which is what they are anyway, being sampled
    seconds apart -- and inference runs only on the sampled frames instead of
    every frame, several times less work.
    """
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=min_score,
            min_tracking_confidence=min_score,
        )
    )


def extract_one(
    landmarker: Any,
    job: dict[str, Any],
    rows: list[dict[str, Any]],
    report: ChunkReport,
) -> None:
    import mediapipe as mp

    capture = cv2.VideoCapture(job["path"])
    if not capture.isOpened():
        report.videos_unreadable.append(job["video_id"])
        return

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wanted = evenly_spaced(total, job["frames_per_video"])
    report.videos += 1

    # grab() walks the stream without decoding a frame into memory; only the
    # sampled frames are retrieved. At 1080x1920 a decoded frame is ~6 MB, so
    # decoding all of them across parallel workers exhausted OpenCV's
    # allocator. Seeking instead of walking is not reliable across the codecs
    # in this dataset. Sampling at all is what stops eight near-identical
    # frames from one clip outweighing a whole other signer.
    index = 0
    while capture.grab():
        if index in wanted:
            ok, frame = capture.retrieve()
            if ok and frame is not None and frame.size:
                # The array must be contiguous and must outlive the detection:
                # mp.Image wraps the buffer rather than copying it, so passing
                # the conversion's result inline -- with nothing holding a
                # reference -- crashed the worker with an access violation.
                rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                record(landmarker.detect(image), job, index, width, height, rows, report)
                del image, rgb, frame
        index += 1

    capture.release()


def record(
    result: Any,
    job: dict[str, Any],
    frame_index: int,
    width: int,
    height: int,
    rows: list[dict[str, Any]],
    report: ChunkReport,
) -> None:
    letter = job["letter"]
    if not result.hand_landmarks:
        report.frames_no_hand[letter] += 1
        return

    category = result.handedness[0][0]
    if category.score < job["min_score"]:
        report.frames_low_score[letter] += 1
        return

    landmarks = result.hand_landmarks[0]
    if len(landmarks) != NUM_LANDMARKS:
        report.frames_no_hand[letter] += 1
        return

    row: dict[str, Any] = {
        "video_id": job["video_id"],
        "signer_id": job["signer_id"],
        "letter": letter,
        "train": job["train"],
        "frame_idx": frame_index,
        "handedness": physical_hand(category.category_name),
        "width": width,
        "height": height,
    }
    for index, point in enumerate(landmarks):
        row[f"x{index}"] = point.x
        row[f"y{index}"] = point.y
        row[f"z{index}"] = point.z

    rows.append(row)
    report.frames_kept[letter] += 1


def physical_hand(reported: str) -> str:
    """Undo MediaPipe's selfie assumption, exactly as landmarker.ts does.

    Its label assumes a horizontally flipped image; these clips are not
    flipped. The browser swaps it, so this must swap it too -- the convention
    only has to be the same in both, and a difference would be invisible until
    the model mysteriously underperformed.
    """
    return "Right" if reported == "Left" else "Left"


def evenly_spaced(total: int, wanted: int) -> set[int]:
    if total <= 0:
        return set()
    if total <= wanted:
        return set(range(total))
    return {int(round(position)) for position in np.linspace(0, total - 1, wanted)}


def merge_shards(args: argparse.Namespace) -> None:
    shards = sorted(args.shards.glob("*.parquet"))
    if not shards:
        raise SystemExit("no shards were produced")

    frame = pd.concat([pd.read_parquet(shard) for shard in shards], ignore_index=True)
    frame = frame.drop_duplicates(subset=["video_id", "frame_idx"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)

    print(f"\n{len(frame)} rows -> {args.out}")
    print(f"  {frame['video_id'].nunique()} videos, {frame['signer_id'].nunique()} signers")
    print(f"  {frame['letter'].nunique()} classes")
    print("  frames per class (lowest first):")
    for letter, count in frame["letter"].value_counts().tail(8).items():
        print(f"    {letter:<10} {count}")


def write_report(args: argparse.Namespace, report: ChunkReport) -> None:
    path = args.out.parent / "extraction_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "videos_processed": report.videos,
        "frames_kept": dict(report.frames_kept),
        "frames_dropped_no_hand": dict(report.frames_no_hand),
        "frames_dropped_low_score": dict(report.frames_low_score),
        "videos_unreadable": report.videos_unreadable,
        "frames_per_video": args.frames_per_video,
        "min_score": args.min_score,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    kept = sum(report.frames_kept.values())
    dropped = sum(report.frames_no_hand.values()) + sum(report.frames_low_score.values())
    print(f"\nkept {kept} frames, dropped {dropped}; report at {path}")
    if report.videos_unreadable:
        print(f"  {len(report.videos_unreadable)} videos could not be opened")

    worst = sorted(report.frames_no_hand.items(), key=lambda item: item[1], reverse=True)[:5]
    if worst:
        print("  most frames lost to 'no hand detected', by class:")
        for letter, count in worst:
            print(f"    {letter:<10} {count}")


if __name__ == "__main__":
    main()
