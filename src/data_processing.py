"""
ETL Pipeline для обработки данных о бриллиантах
Автор: [Ваше Имя]
"""

import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DiamondsDataProcessor:
    """Класс для обработки данных о бриллиантах"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.preprocessing_pipeline = None
        self.feature_names = None

    def load_data(self, path="data/raw/diamonds.csv"):
        """
        Extract: Загрузка данных из CSV
        """
        print(f"Загрузка данных из {path}...")
        df = pd.read_csv(path)
        print(f"Загружено {len(df)} записей, {len(df.columns)} признаков")
        return df

    def validate_data(self, df):
        """
        Валидация данных
        """
        print("Валидация данных...")

        if df.isnull().sum().sum() > 0:
            print(f"Найдено пропусков: {df.isnull().sum().sum()}")
            df = df.dropna()

        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"Найдено дубликатов: {duplicates}")
            df = df.drop_duplicates()

        assert (df["price"] > 0).all(), "Найдены отрицательные цены!"
        assert (df["carat"] > 0).all(), "Найдены отрицательные караты!"

        print(f"Валидация пройдена. Осталось записей: {len(df)}")
        return df

    def clean_data(self, df):
        """
        Transform: Очистка данных от выбросов
        """
        print("Очистка данных от выбросов...")

        df = df[df["carat"] < 3.0]
        df = df[(df["depth"] > 40) & (df["depth"] < 80)]
        df = df[(df["table"] > 40) & (df["table"] < 80)]
        df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)]

        print(f"После очистки: {len(df)} записей")
        return df

    def engineer_features(self, df):
        """
        Создание новых признаков
        """
        print("Создание новых признаков...")

        df["volume"] = df["x"] * df["y"] * df["z"]
        df["density"] = df["carat"] / (df["volume"] + 0.001)
        df["depth_to_width"] = df["depth"] / (df["x"] + 0.001)

        print("Создано 3 новых признака")
        return df

    def preprocess(self, df, fit=True):
        """
        Transform: Предобработка признаков
        """
        print("Предобработка признаков...")

        numeric_features = [
            "carat",
            "depth",
            "table",
            "x",
            "y",
            "z",
            "volume",
            "density",
            "depth_to_width",
        ]
        categorical_features = ["cut", "color", "clarity"]

        X = df[numeric_features + categorical_features].copy()
        y = df["price"].copy()

        if fit:
            numeric_transformer = StandardScaler()
            categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

            self.preprocessing_pipeline = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_features),
                    ("cat", categorical_transformer, categorical_features),
                ]
            )

            X_processed = self.preprocessing_pipeline.fit_transform(X)

            num_feature_names = numeric_features
            cat_feature_names = (
                self.preprocessing_pipeline.named_transformers_["cat"]
                .get_feature_names_out(categorical_features)
                .tolist()
            )
            self.feature_names = num_feature_names + cat_feature_names

            os.makedirs("models", exist_ok=True)
            joblib.dump(self.preprocessing_pipeline, "models/preprocessor.pkl")
            print("Препроцессор сохранен в models/preprocessor.pkl")
        else:
            self.preprocessing_pipeline = joblib.load("models/preprocessor.pkl")
            X_processed = self.preprocessing_pipeline.transform(X)

        print(f"Предобработка завершена. Форма данных: {X_processed.shape}")
        return X_processed, y

    def save_processed_data(self, X, y, X_test, y_test, path="data/processed/"):
        """
        Load: Сохранение обработанных данных
        """
        print("Сохранение обработанных данных...")

        os.makedirs(path, exist_ok=True)

        joblib.dump((X, y), f"{path}/train_data.pkl")
        joblib.dump((X_test, y_test), f"{path}/test_data.pkl")

        print(f"Данные сохранены в {path}")

    def run_full_pipeline(self):
        """
        Запуск полного ETL пайплайна
        """
        print("=" * 60)
        print("ЗАПУСК ETL ПАЙПЛАЙНА")
        print("=" * 60)

        df = self.load_data()
        df = self.validate_data(df)
        df = self.clean_data(df)
        df = self.engineer_features(df)

        os.makedirs("data/processed", exist_ok=True)
        df.to_csv("data/processed/diamonds_cleaned.csv", index=False)
        print("Очищенные данные сохранены")

        X_train, X_test, y_train, y_test = train_test_split(df, df["price"], test_size=0.2, random_state=42)

        X_train_processed, y_train = self.preprocess(X_train, fit=True)
        X_test_processed, y_test = self.preprocess(X_test, fit=False)

        self.save_processed_data(X_train_processed, y_train, X_test_processed, y_test)

        print("=" * 60)
        print("ETL ПАЙПЛАЙН ЗАВЕРШЕН УСПЕШНО!")
        print("=" * 60)

        return X_train_processed, y_train, X_test_processed, y_test


if __name__ == "__main__":
    processor = DiamondsDataProcessor()
    processor.run_full_pipeline()
