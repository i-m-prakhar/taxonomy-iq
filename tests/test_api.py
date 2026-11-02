from fastapi.testclient import TestClient
from taxonomyiq.api import create_app
from taxonomyiq.storage import Store

def test_api_prediction_feedback_and_batch(trained):
    model, db = trained
    with TestClient(create_app(model, db)) as client:
        assert client.get("/health/ready").status_code == 200
        resp = client.post("/v1/predict", json={"description": "stainless steel frying pan"})
        assert resp.status_code == 200, resp.text
        obj = resp.json()
        assert obj["request_id"]
        assert 0 <= obj["confidence"] <= 1
        assert client.post("/v1/feedback", json={"request_id": obj["request_id"], "corrected_category": "home_kitchen"}).status_code == 200
        assert client.post("/v1/feedback", json={"request_id": "missing", "corrected_category": "home_kitchen"}).status_code == 404
        bad = client.post("/v1/predict", json={"description": "  "})
        assert bad.status_code == 422
        batch = client.post("/v1/predict/batch", json={"items": [{"description": "wireless headphones"}, {"description": "natural soap body cleanser"}]})
        assert batch.json()["count"] == 2
        assert client.get("/metrics").status_code == 200
        assert client.get("/v1/monitor/drift").json()["n"] == 3

def test_request_id_idempotency(trained):
    model, db = trained
    with TestClient(create_app(model, db)) as client:
        for _ in range(2):
            assert client.post("/v1/predict", json={"description": "organic whole milk", "request_id": "repeat"}).status_code == 200
    with Store(db).connect() as con:
        assert con.execute("SELECT count(*) FROM predictions WHERE request_id='repeat'").fetchone()[0] == 1
