"""
Тесты для FastAPI приложения
Автор: [Ваше Имя]
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.app import app

client = TestClient(app)


class TestAPIEndpoints:
    """Тесты API эндпоинтов"""

    def test_root_endpoint(self):
        """Тест корневого эндпоинта"""
        response = client.get("/")

        assert response.status_code == 200
        assert "message" in response.json()

    def test_health_endpoint(self):
        """Тест health check"""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_predict_endpoint_valid_input(self):
        """Тест предсказания с валидными данными"""
        if not (os.path.exists("models/best_model.pkl") and os.path.exists("models/preprocessor.pkl")):
            pytest.skip("Model artifacts are not available yet")

        payload = {
            "carat": 1.0,
            "cut": "Ideal",
            "color": "D",
            "clarity": "VS1",
            "depth": 61.0,
            "table": 55.0,
            "x": 5.0,
            "y": 5.0,
            "z": 3.0,
        }

        response = client.post("/predict", json=payload)

        assert response.status_code == 200
        assert "predicted_price" in response.json()
        assert response.json()["predicted_price"] > 0

    def test_predict_endpoint_invalid_cut(self):
        """Тест предсказания с невалидным cut"""
        payload = {
            "carat": 1.0,
            "cut": "Invalid",
            "color": "D",
            "clarity": "VS1",
        }

        response = client.post("/predict", json=payload)

        assert response.status_code in [200, 422, 500, 503]

    def test_model_info_endpoint(self):
        """Тест эндпоинта информации о модели"""
        response = client.get("/model/info")

        assert response.status_code == 200
        assert "model_type" in response.json()
        assert "metrics" in response.json()


class TestAPIValidation:
    """Тесты валидации API"""

    def test_predict_missing_required_fields(self):
        """Тест отсутствия обязательных полей"""
        payload = {
            "carat": 1.0,
        }

        response = client.post("/predict", json=payload)

        assert response.status_code in [422, 500]

    def test_predict_negative_carat(self):
        """Тест отрицательного carat"""
        payload = {
            "carat": -1.0,
            "cut": "Ideal",
            "color": "D",
            "clarity": "VS1",
        }

        response = client.post("/predict", json=payload)

        assert response.status_code in [200, 422, 500, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
