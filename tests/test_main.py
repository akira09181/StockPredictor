from fastapi.testclient import TestClient
from main import app
import os

client = TestClient(app)

if not os.path.exists("static"):
    os.makedirs("static")

def test_predict_plot():
    response = client.post("/predict_adjust", json={
        "ticker": "AAPL",
        "period": "1y",
        "steps": 30
    })
    assert response.status_code == 200
    data = response.json()
    assert "file_path" in data
    assert "train_mae" in data
    assert "train_mse" in data
    assert "train_r2" in data
    assert "test_mae" in data
    assert "test_mse" in data
    assert "test_r2" in data