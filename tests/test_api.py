from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

SAMPLE_PAYLOAD = {
    "age": 52,
    "sex": 1,
    "cp": 0,
    "trestbps": 125,
    "chol": 212,
    "fbs": 0,
    "restecg": 1,
    "thalach": 168,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 2,
    "thal": 3,
}


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_prediction_schema_validation_or_model_response():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code in {200, 503}
    if response.status_code == 200:
        body = response.json()
        assert body["prediction"] in {0, 1}
        assert "confidence" in body
    else:
        assert "Model artifact not found" in response.text
