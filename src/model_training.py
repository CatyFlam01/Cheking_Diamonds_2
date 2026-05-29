"""
AutoML Training Pipeline для предсказания цен на бриллианты
Автор: [Ваше Имя]
"""

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pycaret.regression import compare_models, save_model, setup, tune_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


class DiamondsModelTrainer:
    """Класс для обучения ML модели"""

    def __init__(self):
        self.model = None
        self.results = None
        self.metrics = {}
        self.model_name = None

    def load_data(self):
        """
        Загрузка обработанных данных
        """
        print("Загрузка обработанных данных...")
        X_train, y_train = joblib.load("data/processed/train_data.pkl")
        X_test, y_test = joblib.load("data/processed/test_data.pkl")

        print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, y_train, X_test, y_test

    def train_with_pycaret(self, X_train, y_train):
        """
        AutoML с использованием PyCaret
        """
        print("=" * 60)
        print("ЗАПУСК AUTOML (PyCaret)")
        print("=" * 60)

        train_df = pd.DataFrame(X_train)
        train_df["target"] = y_train

        setup(
            data=train_df,
            target="target",
            train_size=0.8,
            normalize=True,
            transformation=True,
            verbose=False,
        )

        print("Сравнение моделей...")
        best_models = compare_models(n_select=5)

        print("Выбор лучшей модели...")
        best_model = best_models[0]
        model_name = type(best_model).__name__

        print(f"Лучшая модель: {model_name}")

        print("Тюнинг гиперпараметров...")
        tuned_model = tune_model(best_model, n_iter=20)

        print("Сохранение модели...")
        os.makedirs("models", exist_ok=True)
        save_model(tuned_model, "models/best_model")

        self.model = tuned_model
        self.model_name = model_name

        return tuned_model

    def train_with_catboost(self, X_train, y_train, X_test, y_test):
        """
        Альтернативное обучение CatBoost
        """
        from catboost import CatBoostRegressor

        print("=" * 60)
        print("ОБУЧЕНИЕ CatBoost")
        print("=" * 60)

        model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.1,
            depth=10,
            loss_function="RMSE",
            verbose=100,
            random_seed=42,
        )

        model.fit(X_train, y_train, eval_set=(X_test, y_test))

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/catboost_model.pkl")
        joblib.dump(model, "models/best_model.pkl")

        self.model = model
        self.model_name = "CatBoostRegressor"

        return model

    def evaluate_model(self, X_test, y_test):
        """
        Оценка качества модели
        """
        print("=" * 60)
        print("ОЦЕНКА МОДЕЛИ")
        print("=" * 60)

        y_pred = self.model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        self.metrics = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "MAPE": mape,
        }

        print(f"RMSE:  ${rmse:.2f}")
        print(f"MAE:   ${mae:.2f}")
        print(f"R2:    {r2:.4f}")
        print(f"MAPE:  {mape:.2f}%")

        os.makedirs("models", exist_ok=True)
        joblib.dump(self.metrics, "models/metrics.pkl")

        return y_pred

    def create_visualizations(self, y_test, y_pred, X_test):
        """
        Создание визуализаций
        """
        print("=" * 60)
        print("СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
        print("=" * 60)

        os.makedirs("reports/figures", exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5, s=10)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title("Actual vs Predicted Prices")
        plt.tight_layout()
        plt.savefig("reports/figures/actual_vs_predicted.png", dpi=300)
        plt.close()
        print("Saved: actual_vs_predicted.png")

        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5, s=10)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Predicted Price")
        plt.ylabel("Residuals")
        plt.title("Residuals Distribution")
        plt.tight_layout()
        plt.savefig("reports/figures/residuals.png", dpi=300)
        plt.close()
        print("Saved: residuals.png")

        try:
            if hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_
                plt.figure(figsize=(10, 8))
                plt.barh(range(min(10, len(importances))), sorted(importances, reverse=True)[:10])
                plt.xlabel("Importance")
                plt.title("Top 10 Feature Importances")
                plt.tight_layout()
                plt.savefig("reports/figures/feature_importance.png", dpi=300)
                plt.close()
                print("Saved: feature_importance.png")
        except Exception:
            print("Feature importance unavailable")

        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.title("Distribution of Prediction Errors")
        plt.axvline(x=0, color="r", linestyle="--", linewidth=2)
        plt.tight_layout()
        plt.savefig("reports/figures/error_distribution.png", dpi=300)
        plt.close()
        print("Saved: error_distribution.png")

    def log_to_mlflow(self):
        """
        Логирование в MLFlow
        """
        print("=" * 60)
        print("ЛОГИРОВАНИЕ В MLFLOW")
        print("=" * 60)

        mlflow.set_experiment("Diamonds_Price_Prediction")

        with mlflow.start_run():
            mlflow.log_param("model_type", self.model_name)
            mlflow.log_param("dataset", "Kaggle Diamonds")

            mlflow.log_metric("RMSE", self.metrics["RMSE"])
            mlflow.log_metric("MAE", self.metrics["MAE"])
            mlflow.log_metric("R2", self.metrics["R2"])
            mlflow.log_metric("MAPE", self.metrics["MAPE"])

            mlflow.sklearn.log_model(self.model, "model")

            print(f"MLFlow Run ID: {mlflow.active_run().info.run_id}")

    def run_full_training(self):
        """
        Запуск полного пайплайна обучения
        """
        print("=" * 60)
        print("ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА ОБУЧЕНИЯ")
        print("=" * 60)

        X_train, y_train, X_test, y_test = self.load_data()
        self.train_with_catboost(X_train, y_train, X_test, y_test)
        y_pred = self.evaluate_model(X_test, y_test)
        self.create_visualizations(y_test, y_pred, X_test)
        self.log_to_mlflow()

        print("=" * 60)
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("=" * 60)

        print("\nИТОГОВЫЕ МЕТРИКИ:")
        for metric, value in self.metrics.items():
            print(f"   {metric}: {value:.4f}")


if __name__ == "__main__":
    trainer = DiamondsModelTrainer()
    trainer.run_full_training()
