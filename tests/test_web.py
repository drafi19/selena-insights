from pathlib import Path

from fastapi.testclient import TestClient

from selena_ml.serve import app as app_mod


def _client(monkeypatch) -> TestClient:
    monkeypatch.setattr(app_mod, "WEB_DIR", Path(__file__).resolve().parents[1] / "web")
    return TestClient(app_mod.app)


def test_index_serves_web_page(monkeypatch):
    r = _client(monkeypatch).get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_index_markup_is_mobile_safe_and_reports_failures(monkeypatch):
    html = _client(monkeypatch).get("/").text
    assert 'name="viewport"' in html
    assert "chart-wrap" in html
    assert "table-wrap" in html
    assert "--muted" in html
    assert ".error" in html
    assert "'error'" in html
    assert 'height="110"' not in html
