from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


def test_single_classify():
    payload = {"log": "timeout connecting to server"}
    response = client.post("/classify", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "label" in body
    assert "confidence" in body
    assert "layer" in body


def test_batch_csv_upload():
    csv_data = "log\nconnection refused by database\n"
    files = {"file": ("logs.csv", csv_data, "text/csv")}
    response = client.post("/classify_csv", files=files)
    assert response.status_code == 200
    items = response.json()
    assert len(items) == 1
    assert "label" in items[0]
