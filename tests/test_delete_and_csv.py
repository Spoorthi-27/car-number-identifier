"""Tests for DELETE /api/plates/{id} and the CSV download endpoints."""
import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

import backend.api as api_module
import config as config_module


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "DB_PATH", str(tmp_path / "api_tests.db"))
    monkeypatch.setattr(config_module, "SAVE_SNAPSHOTS", False)
    api_module._db = None  # force a fresh DB at the temp path
    yield TestClient(api_module.app)
    if api_module._db is not None:
        api_module._db.close()
    api_module._db = None


def _log_plate(text="MH12AB1234", confidence=0.91):
    db = api_module.get_db()
    db.log_plate(text, confidence=confidence)
    return db.recent_plates(limit=1)[0]["id"]


def _parse_csv(response):
    text = response.content.decode("utf-8-sig")
    return list(csv.reader(io.StringIO(text)))


# ---------------------------------------------------------------- delete
def test_delete_plate_removes_the_row(client):
    plate_id = _log_plate()
    response = client.delete(f"/api/plates/{plate_id}")
    assert response.status_code == 200
    assert response.json() == {"deleted": plate_id}
    assert client.get("/api/plates").json() == []


def test_delete_missing_plate_returns_404(client):
    response = client.delete("/api/plates/99999")
    assert response.status_code == 404


# ------------------------------------------------------------ CSV download
def test_single_plate_download(client):
    plate_id = _log_plate("MH12AB1234")
    response = client.get(f"/api/plates/{plate_id}/download")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert f"MH12AB1234_{plate_id}.csv" in response.headers["content-disposition"]
    assert response.content.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM for Excel

    rows = _parse_csv(response)
    assert rows[0][0] == "Vehicle Number"
    assert len(rows) == 2  # header + one record
    assert rows[1][0] == "MH12AB1234"


def test_single_plate_download_missing_returns_404(client):
    assert client.get("/api/plates/99999/download").status_code == 404


def test_history_download_has_one_row_per_plate(client):
    _log_plate("MH12AB1234")
    _log_plate("DL8CAF5030")
    response = client.get("/api/plates/download")
    assert response.status_code == 200
    assert "plate_history.csv" in response.headers["content-disposition"]

    rows = _parse_csv(response)
    assert len(rows) == 3  # header + two records
    assert {rows[1][0], rows[2][0]} == {"MH12AB1234", "DL8CAF5030"}


def test_history_download_with_no_data_is_header_only(client):
    rows = _parse_csv(client.get("/api/plates/download"))
    assert len(rows) == 1
    assert rows[0][0] == "Vehicle Number"
# ------------------------------------------------------- hardening checks
def test_csv_safe_neutralises_formulas():
    assert api_module._csv_safe("=HYPERLINK(1)") == "'=HYPERLINK(1)"
    assert api_module._csv_safe("+1") == "'+1"
    assert api_module._csv_safe("-1") == "'-1"
    assert api_module._csv_safe("@SUM(A1)") == "'@SUM(A1)"
    assert api_module._csv_safe("MH12AB1234") == "MH12AB1234"
    assert api_module._csv_safe(None) == ""


def test_history_download_limit_is_bounded(client):
    assert client.get("/api/plates/download?limit=0").status_code == 422
    assert client.get("/api/plates/download?limit=5000").status_code == 422
    assert client.get("/api/plates/download?limit=10").status_code == 200


def test_filename_part_is_ascii_only():
    assert api_module._safe_filename_part("é12AB", "vehicle") == "12AB"
    assert api_module._safe_filename_part("", "vehicle") == "vehicle"
    
def test_plates_list_limit_is_bounded(client):
    assert client.get("/api/plates?limit=0").status_code == 422
    assert client.get("/api/plates?limit=5000").status_code == 422
    assert client.get("/api/plates?limit=10").status_code == 200