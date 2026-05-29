"""
Мониторинг инфраструктуры
Автор: [Ваше Имя]
"""

import json
import os
import time
from datetime import datetime

import docker
import psutil


class InfrastructureMonitor:
    """Класс для мониторинга инфраструктуры"""

    def __init__(self):
        self.client = docker.from_env()
        self.metrics_log = []

    def get_system_metrics(self):
        """
        Получение метрик системы
        """
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_mb": memory.available / (1024 * 1024),
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3),
        }

    def get_docker_metrics(self):
        """
        Получение метрик Docker контейнеров
        """
        containers = self.client.containers.list()
        container_metrics = []

        for container in containers:
            if "diamonds" in container.name:
                stats = container.stats(stream=False)

                cpu_delta = (
                    stats["cpu_stats"]["cpu_usage"]["total_usage"]
                    - stats["precpu_stats"]["cpu_usage"]["total_usage"]
                )
                system_delta = stats["cpu_stats"]["system_cpu_usage"] - stats["precpu_stats"]["system_cpu_usage"]

                cpu_percent = (cpu_delta / system_delta) * stats["cpu_stats"]["online_cpus"] * 100

                memory_usage = stats["memory_stats"]["usage"] / (1024 * 1024)
                memory_limit = stats["memory_stats"]["limit"] / (1024 * 1024)

                container_metrics.append(
                    {
                        "container_name": container.name,
                        "status": container.status,
                        "cpu_percent": round(cpu_percent, 2),
                        "memory_usage_mb": round(memory_usage, 2),
                        "memory_limit_mb": round(memory_limit, 2),
                    }
                )

        return container_metrics

    def check_resource_limits(self):
        """
        Проверка соблюдения лимитов ресурсов
        """
        system_metrics = self.get_system_metrics()

        alerts = []

        if system_metrics["cpu_percent"] > 80:
            alerts.append(
                {
                    "severity": "WARNING",
                    "metric": "CPU",
                    "value": system_metrics["cpu_percent"],
                    "threshold": 80,
                    "message": "High CPU usage detected",
                }
            )

        if system_metrics["memory_percent"] > 85:
            alerts.append(
                {
                    "severity": "CRITICAL",
                    "metric": "Memory",
                    "value": system_metrics["memory_percent"],
                    "threshold": 85,
                    "message": "High memory usage detected",
                }
            )

        if system_metrics["disk_percent"] > 90:
            alerts.append(
                {
                    "severity": "CRITICAL",
                    "metric": "Disk",
                    "value": system_metrics["disk_percent"],
                    "threshold": 90,
                    "message": "Low disk space",
                }
            )

        return alerts

    def create_monitoring_report(self):
        """
        Создание отчета о мониторинге
        """
        os.makedirs("reports/monitoring", exist_ok=True)

        system_metrics = self.get_system_metrics()
        docker_metrics = self.get_docker_metrics()
        alerts = self.check_resource_limits()

        report = {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": system_metrics,
            "docker_metrics": docker_metrics,
            "alerts": alerts,
            "status": "OK" if len(alerts) == 0 else "WARNING",
        }

        report_file = f"reports/monitoring/infrastructure_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print("=" * 60)
        print("INFRASTRUCTURE MONITORING REPORT")
        print("=" * 60)
        print(f"Timestamp: {report['timestamp']}")
        print("\nSystem Metrics:")
        print(f"  CPU: {system_metrics['cpu_percent']}%")
        print(f"  Memory: {system_metrics['memory_percent']}%")
        print(f"  Disk: {system_metrics['disk_percent']}%")

        print("\nDocker Containers:")
        for container in docker_metrics:
            print(f"  {container['container_name']}:")
            print(f"    Status: {container['status']}")
            print(f"    CPU: {container['cpu_percent']}%")
            print(f"    Memory: {container['memory_usage_mb']} MB")

        if alerts:
            print(f"\nALERTS ({len(alerts)}):")
            for alert in alerts:
                print(f"  [{alert['severity']}] {alert['metric']}: {alert['message']}")
        else:
            print("\nAll systems operational")

        print("=" * 60)

        return report

    def start_continuous_monitoring(self, interval=60):
        """
        Запуск непрерывного мониторинга
        """
        print(f"Starting continuous monitoring (interval: {interval}s)")
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                self.create_monitoring_report()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped")


if __name__ == "__main__":
    monitor = InfrastructureMonitor()
    monitor.create_monitoring_report()
