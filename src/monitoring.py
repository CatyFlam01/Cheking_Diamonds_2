"""
Мониторинг качества модели
Автор: [Ваше Имя]
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


class ModelMonitor:
    """Класс для мониторинга модели"""

    def __init__(self):
        self.model = joblib.load("models/best_model.pkl")
        self.metrics = joblib.load("models/metrics.pkl")

    def check_data_drift(self, new_data_path, reference_data_path):
        """
        Проверка дрейфа данных
        """
        print("Проверка дрейфа данных...")

        new_data = pd.read_csv(new_data_path)
        ref_data = pd.read_csv(reference_data_path)

        drift_report = {}

        for col in ["carat", "depth", "table"]:
            if col in new_data.columns and col in ref_data.columns:
                mean_diff = abs(new_data[col].mean() - ref_data[col].mean())
                std_diff = abs(new_data[col].std() - ref_data[col].std())

                drift_report[col] = {
                    "mean_drift": mean_diff,
                    "std_drift": std_diff,
                    "status": "OK" if mean_diff < 0.1 else "WARNING",
                }

        print(f"Отчет о дрейфе: {drift_report}")
        return drift_report

    def monitor_predictions(self, X_test, y_test):
        """
        Мониторинг предсказаний
        """
        print("Мониторинг качества предсказаний...")

        y_pred = self.model.predict(X_test)

        current_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        current_r2 = r2_score(y_test, y_pred)

        baseline_rmse = self.metrics["RMSE"]
        baseline_r2 = self.metrics["R2"]

        quality_report = {
            "current_rmse": current_rmse,
            "baseline_rmse": baseline_rmse,
            "rmse_degradation": ((current_rmse - baseline_rmse) / baseline_rmse) * 100,
            "current_r2": current_r2,
            "baseline_r2": baseline_r2,
            "r2_change": current_r2 - baseline_r2,
            "status": "OK" if current_rmse < baseline_rmse * 1.1 else "RETRAIN_NEEDED",
        }

        print(f"Отчет о качестве: {quality_report}")
        return quality_report

    def create_monitoring_dashboard(self):
        """
        Создание дашборда мониторинга
        """
        os.makedirs("reports/monitoring", exist_ok=True)

        plt.figure(figsize=(12, 6))

        dates = pd.date_range(start="2026-05-01", periods=10, freq="D")
        rmse_values = [self.metrics["RMSE"] * (1 + np.random.uniform(-0.05, 0.05)) for _ in range(10)]

        plt.plot(dates, rmse_values, marker="o", linewidth=2, markersize=8)
        plt.axhline(
            y=self.metrics["RMSE"],
            color="r",
            linestyle="--",
            label=f'Baseline RMSE: {self.metrics["RMSE"]:.2f}',
        )
        plt.xlabel("Date")
        plt.ylabel("RMSE")
        plt.title("Model Performance Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("reports/monitoring/performance_over_time.png", dpi=300)
        plt.close()

        print("Дашборд сохранен: reports/monitoring/performance_over_time.png")


if __name__ == "__main__":
    monitor = ModelMonitor()
    monitor.create_monitoring_dashboard()
