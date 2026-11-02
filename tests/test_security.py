from fastapi.testclient import TestClient
import taxonomyiq.api as api

def test_api_key_required(monkeypatch, trained):
    model, db = trained
    monkeypatch.setattr(api, "API_KEY", "test-key")
    with TestClient(api.create_app(model, db)) as client:
        assert client.post("/v1/predict", json={"description": "wireless bluetooth speaker"}).status_code == 401
        assert client.post("/v1/predict", headers={"X-API-Key": "bad"}, json={"description": "wireless bluetooth speaker"}).status_code == 401
        assert client.post("/v1/predict", headers={"X-API-Key": "test-key"}, json={"description": "wireless bluetooth speaker"}).status_code == 200
