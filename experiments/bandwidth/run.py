"""Bandwidth: what the browser sends per frame, against sending JPEG frames.

CLAUDE.md 16.1. Everything is measured offline from real data:

* Landmark messages are rebuilt byte for byte as ws-client.ts sends them, from
  real landmarks in ml/data/bukva_landmarks.parquet. Numbers are formatted by
  ECMAScript's Number-to-string rules, not Python's, because the two disagree
  on small values (JS "0.000025", Python "2.5e-05").
* Server replies are produced by the real pipeline -- the trained model, the
  hint templates and the production EvalOut serializer -- not estimated.
* JPEG sizes come from encoding real Bukva video frames with OpenCV at several
  sizes and qualities. No image is ever sent to the app (CLAUDE.md 7).

Each configuration is run on three independent random samples (seeds 0, 1,
2): encoding is deterministic, so repeating one sample would only repeat its
numbers; different samples show how much the result depends on which frames
were picked.

Usage (from the repository root, with BUKVA_DIR set):
    uv run --project backend --group ml python experiments/bandwidth/run.py
"""

from __future__ import annotations

import io
import json
import os
import platform
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
import dataset  # noqa: E402

from app.application.evaluate_gesture import PracticeSession  # noqa: E402
from app.domain.gesture.features import Handedness, aspect_of  # noqa: E402
from app.domain.gesture.feedback import LetterTemplate  # noqa: E402
from app.infrastructure.ml.model import JoblibClassifier  # noqa: E402
from app.presentation.ws.schemas import EvalOut  # noqa: E402

RESULTS = Path(__file__).with_name("results")
MODEL = ROOT / "ml" / "artifacts" / "model-v2.joblib"
TEMPLATES = ROOT / "ml" / "artifacts" / "templates-v2.json"

SEEDS = (0, 1, 2)
LANDMARK_SAMPLES = 1000
JPEG_VIDEOS = 60
JPEG_LONG_SIDES = (640, 1280)
JPEG_QUALITIES = (50, 75, 90)
#: Date.now() has 13 digits for the rest of this century.
T_CLIENT = 1_792_000_000_000


def js_number(value: float) -> str:
    """ECMAScript Number::toString for a finite double.

    Shortest round-trip digits are the same in both languages (Python's repr
    and JavaScript both produce them); only the layout differs.
    """
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    _, digit_tuple, exponent = Decimal(repr(abs(value))).normalize().as_tuple()
    assert isinstance(exponent, int)
    digits = "".join(map(str, digit_tuple))
    k, n = len(digits), len(digits) + exponent  # value = 0.digits x 10^n

    if k <= n <= 21:
        return sign + digits + "0" * (n - k)
    if 0 < n <= 21:
        return sign + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * (-n) + digits
    e = n - 1
    mantissa = digits if k == 1 else digits[0] + "." + digits[1:]
    return f"{sign}{mantissa}e{'+' if e > 0 else '-'}{abs(e)}"


def frame_message(landmarks: list[list[float]], handedness: str, seq: int) -> str:
    """The exact text ws-client.ts sendFrame() produces (JSON.stringify,
    keys in the same order, no whitespace)."""
    points = ",".join("[" + ",".join(js_number(c) for c in point) + "]" for point in landmarks)
    return (
        f'{{"type":"frame","seq":{seq},"t_client":{T_CLIENT + seq * 67},'
        f'"handedness":"{handedness}","mirrored":false,"landmarks":[{points}]}}'
    )


def ws_frame_bytes(payload: int, *, masked: bool) -> int:
    """RFC 6455 framing: 2-byte header, extended length, 4-byte client mask."""
    header = 2 + (0 if payload < 126 else 2 if payload < 65536 else 8)
    return payload + header + (4 if masked else 0)


def measure_landmarks(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    classifier = JoblibClassifier(MODEL)
    templates = json.loads(TEMPLATES.read_text(encoding="utf-8"))["letters"]
    rows = frame.sample(LANDMARK_SAMPLES, random_state=seed)
    out = []

    for seq, record in enumerate(rows.to_dict("records")):
        letter, hand = str(record["letter"]), str(record["handedness"])
        width, height = int(record["width"]), int(record["height"])
        landmarks = [[float(record[f"{axis}{i}"]) for axis in "xyz"] for i in range(21)]
        message = frame_message(landmarks, hand, seq)
        rounded = frame_message([[round(c, 4) for c in point] for point in landmarks], hand, seq)

        # The learner is attempting the letter they are showing, as in Learn
        # mode; a fresh session per frame keeps hold state out of it.
        entry = templates[letter]
        session = PracticeSession(
            target=letter,
            aspect=aspect_of(width, height),
            classifier=classifier,
            template=LetterTemplate(letter, entry["angles"], entry["tolerances"]),
            confidence_threshold=0.7,
            hold_frames=10,
        )
        evaluation = session.evaluate(landmarks, Handedness(hand), mirrored=False)
        reply = EvalOut.of(
            evaluation, seq=seq, t_client=T_CLIENT, server_ms=0.512
        ).model_dump_json()

        out.append(
            {
                "seed": seed,
                "letter": letter,
                "frame_payload": len(message.encode("utf-8")),
                "frame_wire": ws_frame_bytes(len(message.encode("utf-8")), masked=True),
                "frame_payload_rounded4": len(rounded.encode("utf-8")),
                "reply_payload": len(reply.encode("utf-8")),
                "reply_wire": ws_frame_bytes(len(reply.encode("utf-8")), masked=False),
                "reply_hints": len(evaluation.hints),
            }
        )
    return pd.DataFrame(out)


def measure_jpeg(annotations: pd.DataFrame, bukva: Path, seed: int) -> pd.DataFrame:
    videos = annotations.sample(JPEG_VIDEOS, random_state=seed)
    out = []
    for video in videos.itertuples(index=False):
        capture = cv2.VideoCapture(str(bukva / "trimmed" / f"{video.attachment_id}.mp4"))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
        ok, image = capture.read()
        capture.release()
        if not ok or image is None:
            continue

        height, width = image.shape[:2]
        for long_side in JPEG_LONG_SIDES:
            scale = long_side / max(height, width)
            resized = cv2.resize(
                image,
                (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
            for quality in JPEG_QUALITIES:
                ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
                assert ok
                out.append(
                    {
                        "seed": seed,
                        "video": video.attachment_id,
                        "long_side": long_side,
                        "width": resized.shape[1],
                        "height": resized.shape[0],
                        "quality": quality,
                        "jpeg_bytes": int(encoded.size),
                    }
                )
    return pd.DataFrame(out)


def environment(bukva: Path) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "dataset_sha256": dataset._sha256(dataset.DEFAULT_PARQUET),
        "bukva_dir": str(bukva),
        "model": MODEL.name,
        "seeds": list(SEEDS),
        "landmark_samples_per_seed": LANDMARK_SAMPLES,
        "jpeg_videos_per_seed": JPEG_VIDEOS,
        "note": "offline measurement; no network involved",
    }


def main() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    bukva = Path(os.environ.get("BUKVA_DIR", ""))
    if not (bukva / "annotations.tsv").is_file():
        raise SystemExit("set BUKVA_DIR to the unpacked Bukva dataset")

    letters = set(dataset.load_classes())
    landmarks = pd.read_parquet(dataset.DEFAULT_PARQUET)
    landmarks = landmarks[landmarks["letter"].isin(letters)]
    annotations = pd.read_csv(bukva / "annotations.tsv", sep="\t")
    annotations = annotations[annotations["text"].isin(letters)]

    RESULTS.mkdir(exist_ok=True)
    landmark_runs = []
    jpeg_runs = []
    for seed in SEEDS:
        print(f"seed {seed}: landmark messages ...")
        landmark_runs.append(measure_landmarks(landmarks, seed))
        print(f"seed {seed}: JPEG frames ...")
        jpeg_runs.append(measure_jpeg(annotations, bukva, seed))

    pd.concat(landmark_runs).to_csv(RESULTS / "landmarks.csv", index=False)
    pd.concat(jpeg_runs).to_csv(RESULTS / "jpeg.csv", index=False)
    (RESULTS / "environment.json").write_text(
        json.dumps(environment(bukva), indent=2), encoding="utf-8"
    )
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
