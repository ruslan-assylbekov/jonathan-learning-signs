"""Templates are validated before they can reach the database."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.domain.gesture.features import FEATURE_VERSION, FINGERS, JOINTS
from app.infrastructure.db.load_templates import TemplateFileError, read


def entry() -> dict[str, Any]:
    return {
        "angles": {finger: dict.fromkeys(JOINTS, 20.0) for finger in FINGERS},
        "tolerances": dict.fromkeys(FINGERS, 30.0),
    }


def write(tmp_path: Path, **overrides: Any) -> Path:
    payload = {"feature_version": FEATURE_VERSION, "letters": {"А": entry()}} | overrides
    path = tmp_path / "templates.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_a_valid_file_is_read(tmp_path: Path) -> None:
    letters = read(write(tmp_path))

    assert set(letters) == {"А"}


def test_templates_for_another_feature_version_are_refused(tmp_path: Path) -> None:
    """Angles measured under other normalization would mislead every hint."""
    with pytest.raises(TemplateFileError, match="feature version"):
        read(write(tmp_path, feature_version=FEATURE_VERSION - 1))


def test_a_malformed_template_is_refused_before_any_write(tmp_path: Path) -> None:
    broken = entry()
    del broken["tolerances"]["pinky"]

    with pytest.raises(ValueError, match="tolerance.pinky"):
        read(write(tmp_path, letters={"А": entry(), "Б": broken}))


def test_the_shipped_templates_are_valid() -> None:
    shipped = Path(__file__).resolve().parents[3] / "ml" / "artifacts" / "templates-v2.json"
    if not shipped.is_file():
        pytest.skip("no templates artifact in this checkout")

    assert len(read(shipped)) == 25
