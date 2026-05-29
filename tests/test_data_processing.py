"""
Тесты для модуля обработки данных
Автор: [Ваше Имя]
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processing import DiamondsDataProcessor


class TestDataValidation:
    """Тесты валидации данных"""

    @pytest.fixture
    def sample_df(self):
        """Создание тестового DataFrame"""
        return pd.DataFrame(
            {
                "carat": [0.5, 1.0, 1.5, 2.0],
                "cut": ["Ideal", "Premium", "Good", "Very Good"],
                "color": ["D", "E", "F", "G"],
                "clarity": ["VS1", "VS2", "SI1", "SI2"],
                "depth": [61.0, 62.0, 63.0, 64.0],
                "table": [55.0, 56.0, 57.0, 58.0],
                "x": [5.0, 5.5, 6.0, 6.5],
                "y": [5.0, 5.5, 6.0, 6.5],
                "z": [3.0, 3.5, 4.0, 4.5],
                "price": [1000, 2000, 3000, 4000],
            }
        )

    def test_no_null_values(self, sample_df):
        """Проверка отсутствия пропусков"""
        processor = DiamondsDataProcessor()
        df_clean = processor.validate_data(sample_df)

        assert df_clean.isnull().sum().sum() == 0, "DataFrame содержит пропуски"

    def test_positive_prices(self, sample_df):
        """Проверка положительных цен"""
        processor = DiamondsDataProcessor()
        df_clean = processor.validate_data(sample_df)

        assert (df_clean["price"] > 0).all(), "Найдены отрицательные цены"

    def test_positive_carat(self, sample_df):
        """Проверка положительных каратов"""
        processor = DiamondsDataProcessor()
        df_clean = processor.validate_data(sample_df)

        assert (df_clean["carat"] > 0).all(), "Найдены отрицательные караты"


class TestDataCleaning:
    """Тесты очистки данных"""

    @pytest.fixture
    def df_with_outliers(self):
        """DataFrame с выбросами"""
        return pd.DataFrame(
            {
                "carat": [0.5, 1.0, 3.5, 5.0],
                "cut": ["Ideal", "Premium", "Good", "Very Good"],
                "color": ["D", "E", "F", "G"],
                "clarity": ["VS1", "VS2", "SI1", "SI2"],
                "depth": [61.0, 62.0, 85.0, 30.0],
                "table": [55.0, 56.0, 57.0, 58.0],
                "x": [5.0, 5.5, 6.0, 0.0],
                "y": [5.0, 5.5, 6.0, 6.5],
                "z": [3.0, 3.5, 4.0, 4.5],
                "price": [1000, 2000, 3000, 4000],
            }
        )

    def test_remove_carat_outliers(self, df_with_outliers):
        """Проверка удаления выбросов по carat"""
        processor = DiamondsDataProcessor()
        df_clean = processor.clean_data(df_with_outliers)

        assert (df_clean["carat"] < 3.0).all(), "Выбросы по carat не удалены"

    def test_remove_depth_outliers(self, df_with_outliers):
        """Проверка удаления выбросов по depth"""
        processor = DiamondsDataProcessor()
        df_clean = processor.clean_data(df_with_outliers)

        assert ((df_clean["depth"] > 40) & (df_clean["depth"] < 80)).all(), "Выбросы по depth не удалены"


class TestFeatureEngineering:
    """Тесты создания признаков"""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "carat": [1.0, 2.0],
                "cut": ["Ideal", "Premium"],
                "color": ["D", "E"],
                "clarity": ["VS1", "VS2"],
                "depth": [61.0, 62.0],
                "table": [55.0, 56.0],
                "x": [5.0, 6.0],
                "y": [5.0, 6.0],
                "z": [3.0, 4.0],
                "price": [2000, 4000],
            }
        )

    def test_volume_creation(self, sample_df):
        """Проверка создания признака volume"""
        processor = DiamondsDataProcessor()
        df_engineered = processor.engineer_features(sample_df)

        assert "volume" in df_engineered.columns, "Признак volume не создан"
        assert df_engineered["volume"].iloc[0] == 75.0, "Неверный расчет volume"

    def test_density_creation(self, sample_df):
        """Проверка создания признака density"""
        processor = DiamondsDataProcessor()
        df_engineered = processor.engineer_features(sample_df)

        assert "density" in df_engineered.columns, "Признак density не создан"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
