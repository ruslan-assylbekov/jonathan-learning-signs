from fastapi.testclient import TestClient

from app.domain.gesture.features import FEATURE_VERSION
from app.main import app


def test_health_reports_what_is_running() -> None:
    """The feature version and the stub flag are the two things worth knowing
    about a deployment at a glance (CLAUDE.md 6.3)."""
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["feature_version"] == FEATURE_VERSION
    assert body["stub"] is True, "a real model has not been wired yet"
