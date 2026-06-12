from fastapi.testclient import TestClient

from app.main import create_app


def test_health_returns_ok():
    with TestClient(create_app()) as client:
        res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}
