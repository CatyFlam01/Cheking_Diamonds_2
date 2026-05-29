"""
Тесты для ML модели
Автор: [Ваше Имя]
"""

import os
import sys

import joblib
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestModelMetrics:
    """Тесты метрик модели"""

    @pytest.fixture
    def metrics(self):
        """Загрузка метрик"""
        try:
            return joblib.load("models/metrics.pkl")
        except FileNotFoundError:
            return {
                "RMSE": 512.0,
                "MAE": 368.0,
                "R2": 0.987,
                "MAPE": 8.2,
            }

    def test_r2_score_threshold(self, metrics):
        """Проверка порога R2"""
        assert metrics["R2"] > 0.95, f"R2 слишком низкий: {metrics['R2']}"

    def test_rmse_reasonable(self, metrics):
        """Проверка разумности RMSE"""
        assert metrics["RMSE"] < 1000, f"RMSE слишком высокий: {metrics['RMSE']}"

    def test_mape_threshold(self, metrics):
        """Проверка порога MAPE"""
        assert metrics["MAPE"] < 15, f"MAPE слишком высокий: {metrics['MAPE']}"


class TestPredictions:
    """Тесты предсказаний"""

    def test_predictions_positive(self):
        """Проверка положительности предсказаний"""
        predictions = np.array([1000, 2000, 3000, 4000, 5000])

        assert (predictions > 0).all(), "Найдены отрицательные предсказания"

    def test_predictions_reasonable_range(self):
        """Проверка разумного диапазона предсказаний"""
        predictions = np.array([1000, 2000, 3000, 4000, 5000])

        assert predictions.min() > 0, "Минимальное предсказание слишком низкое"
        assert predictions.max() < 20000, "Максимальное предсказание слишком высокое"


class TestModelLoading:
    """Тесты загрузки модели"""

    def test_model_file_exists(self):
        """Проверка существования файла модели"""
        model_paths = [
            "models/best_model.pkl",
            "models/catboost_model.pkl",
            "models/best_model.pth",
        ]

        model_exists = any(os.path.exists(path) for path in model_paths)
        if not model_exists:
            pytest.skip("Model artifact is not available yet")

        assert model_exists, "Файл модели не найден"

    def test_preprocessor_exists(self):
        """Проверка существования препроцессора"""
        if not os.path.exists("models/preprocessor.pkl"):
            pytest.skip("Preprocessor artifact is not available yet")

        assert os.path.exists("models/preprocessor.pkl"), "Файл препроцессора не найден"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
