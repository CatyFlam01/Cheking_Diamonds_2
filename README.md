# Automation_ML_Diamonds

Команда: 4 человека  
Датасет: Kaggle Diamonds https://www.kaggle.com/datasets/shivam2503/diamonds  
Статус: В работе  
Репозиторий GitHub: https://github.com/W1llAnn/Automation_ML_Diamonds  
Доступ преподавателю: @ElenaSmyslovskikh  

**ML-система предсказания стоимости бриллиантов с MLOps практиками**

## Команда и распределение ролей

| Условная роль | Основные задачи | Вклад в проект |
| --- | --- | --- |
| Data Engineer | ETL-пайплайн, обработка данных, бизнес-логика, API (`app.py`) | Описание проекта, ETL, бизнес-задача |
| ML Engineer | AutoML, обучение модели, метрики, визуализация, мониторинг модели | AutoML, мониторинг модели |
| DevOps Engineer | Docker, контейнеризация, инфраструктура, мониторинг ресурсов | Docker, мониторинг инфраструктуры |
| MLOps & QA | Тестирование (`pytest`), CI/CD, Git-flow, сборка презентации | Тесты, CI/CD, презентация, GitHub |

## Описание бизнес-задачи

Проблема:  
Ручная оценка бриллиантов требует времени и экспертизы. Компания теряет клиентов из-за медленного расчета стоимости.

Цель:  
Создать автоматизированную систему предсказания цены на основе характеристик камня (`carat`, `cut`, `color`, `clarity`, `depth`, `table`, `x`, `y`, `z`).

Ожидаемый эффект:

- Сокращение времени оценки с 15 минут до 1 секунды.
- Унификация цен для клиентов.
- Повышение прозрачности ценообразования.
- Возможность дальнейшей интеграции модели в CRM, интернет-магазин или внутреннюю систему оценки.

## Описание датасета

Используется публичный датасет **Kaggle Diamonds**:  
https://www.kaggle.com/datasets/shivam2503/diamonds

Датасет содержит почти 54 000 записей о бриллиантах и их характеристиках. Целевая переменная проекта - `price`, цена бриллианта в долларах США.

| Признак | Описание | Диапазон / категории |
| --- | --- | --- |
| `price` | Цена в долларах США | $326 - $18,823 |
| `carat` | Вес бриллианта | 0.2 - 5.01 |
| `cut` | Качество огранки | Fair, Good, Very Good, Premium, Ideal |
| `color` | Цвет бриллианта | J - худший, D - лучший |
| `clarity` | Чистота бриллианта | I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF |
| `x` | Длина в мм | 0 - 10.74 |
| `y` | Ширина в мм | 0 - 58.9 |
| `z` | Глубина в мм | 0 - 31.8 |
| `depth` | Общая глубина в процентах | 43 - 79 |
| `table` | Ширина верхней площадки относительно самой широкой точки | 43 - 95 |

Формула признака `depth` в описании датасета:

```text
depth = 2 * z / (x + y)
```

## Ожидаемые метрики

| Метрика | Цель | Факт |
| --- | --- | --- |
| R2 Score | > 0.95 | - |
| RMSE | < $600 | - |
| MAE | Ниже базовой модели | - |
| MAPE | < 15% | - |
| Покрытие тестов | > 70% | - |
| Uptime API | > 99% | - |

## Структура репозитория

```text
diamonds-project/
├── data/
│   ├── raw/                         # Data Engineer: сырые данные
│   └── processed/                   # Data Engineer: обработанные данные
├── src/
│   ├── data_processing.py           # Data Engineer: ETL пайплайн
│   ├── model_training.py            # ML Engineer: обучение модели
│   ├── monitoring.py                # ML Engineer: мониторинг модели
│   ├── app.py                       # Data Engineer: FastAPI приложение
│   └── infrastructure_monitoring.py # DevOps Engineer: мониторинг ресурсов
├── tests/
│   ├── test_data_processing.py      # MLOps & QA: тесты данных
│   ├── test_model.py                # MLOps & QA: тесты модели
│   └── test_api.py                  # MLOps & QA: тесты API
├── docker/
│   ├── Dockerfile                   # DevOps Engineer: образ контейнера
│   ├── prometheus.yml               # DevOps Engineer: конфиг мониторинга
│   └── monitor_resources.sh         # DevOps Engineer: мониторинг Docker
├── .github/workflows/
│   └── ci-cd.yml                    # MLOps & QA: CI/CD пайплайн
├── models/                          # ML Engineer: сохранённые модели
├── reports/
│   ├── figures/                     # ML Engineer: визуализации
│   └── monitoring/                  # DevOps Engineer: дашборды
├── presentation/                    # MLOps & QA: слайды
├── requirements.txt                 # Все: зависимости
├── docker-compose.yml               # DevOps Engineer: запуск сервисов
├── .dockerignore                    # DevOps Engineer: исключения Docker
├── .gitignore                       # MLOps & QA: исключения Git
└── README.md                        # Все: документация
```

## Архитектура ML-системы

### ETL Pipeline

```text
Raw Data -> Validation -> Cleaning -> Feature Engineering -> Preprocessing
```

ETL-пайплайн реализован в `src/data_processing.py` и включает:

- Extract: загрузку CSV-файла `data/raw/diamonds.csv`.
- Validation: проверку пропусков, дубликатов, положительных значений `price` и `carat`.
- Cleaning: удаление выбросов по `carat`, `depth`, `table`, а также строк с нулевыми размерами `x`, `y`, `z`.
- Feature Engineering: создание признаков `volume`, `density`, `depth_to_width`.
- Load: сохранение очищенных и обработанных данных в `data/processed/`.

### ML Pipeline

```text
Processed Data -> AutoML / CatBoost -> Model Selection -> Evaluation -> MLflow Logging -> Deployment
```

Обучение модели реализовано в `src/model_training.py`.

Используемые подходы:

- AutoML через PyCaret.
- Альтернативная кастомная модель `CatBoostRegressor`.
- Оценка качества через RMSE, MAE, R2, MAPE.
- Сохранение модели в `models/best_model.pkl`.
- Логирование экспериментов в MLflow.

### API

FastAPI-приложение находится в `src/app.py`.

Основные эндпоинты:

- `GET /` - информация о сервисе.
- `GET /health` - health check.
- `POST /predict` - предсказание цены бриллианта.
- `GET /model/info` - информация о модели.

Если модель и препроцессор ещё не обучены и не сохранены, `/predict` возвращает сервисную ошибку `503`, потому что артефакты `models/best_model.pkl` и `models/preprocessor.pkl` отсутствуют.

## Детальные задачи по участникам

### Data Engineer

Ветка: `feature/member1-etl`  
Зависимости: нет, начинает первым.

| № | Задача | Файл |
| --- | --- | --- |
| 1.1 | Скачать датасет с Kaggle | `data/raw/diamonds.csv` |
| 1.2 | Создать ETL пайплайн: загрузка, валидация, очистка | `src/data_processing.py` |
| 1.3 | Реализовать Feature Engineering: `volume`, `density`, `depth_to_width` | `src/data_processing.py` |
| 1.4 | Создать FastAPI приложение для предсказаний | `src/app.py` |
| 1.5 | Сохранить обработанные данные для ML Engineer | `data/processed/train_data.pkl`, `data/processed/test_data.pkl` |
| 1.6 | Написать требования к данным для тестов MLOps & QA | `docs/data_spec.md` |

Ожидаемый результат:

- Рабочий ETL пайплайн.
- Обработанные данные в `data/processed/`.
- API с эндпоинтом `/predict`.
- Файл `requirements.txt` с зависимостями.

От чего зависят другие:

- ML Engineer ждёт обработанные данные для обучения.
- MLOps & QA ждёт API для написания тестов.
- DevOps Engineer ждёт `app.py` для контейнеризации.

### ML Engineer

Ветка: `feature/member2-model`  
Зависимости: Data Engineer, обработанные данные.

| № | Задача | Файл |
| --- | --- | --- |
| 2.1 | Загрузить обработанные данные от Data Engineer | `data/processed/` |
| 2.2 | Реализовать AutoML через PyCaret или кастомную модель CatBoost | `src/model_training.py` |
| 2.3 | Обучить модель и сохранить | `models/best_model.pkl` |
| 2.4 | Рассчитать метрики RMSE, MAE, R2, MAPE | `models/metrics.pkl` |
| 2.5 | Создать визуализации: actual vs predicted, residuals, feature importance, error distribution | `reports/figures/` |
| 2.6 | Реализовать мониторинг качества модели | `src/monitoring.py` |
| 2.7 | Настроить MLflow для логирования экспериментов | `mlruns/` |

Ожидаемый результат:

- Обученная модель с R2 > 0.95.
- 4 визуализации в `reports/figures/`.
- Файл метрик `models/metrics.pkl`.
- Скрипт мониторинга дрейфа данных.

От чего зависят другие:

- DevOps Engineer ждёт модель для включения в Docker образ.
- MLOps & QA ждёт метрики для тестов модели.
- Все используют графики для презентации.

### DevOps Engineer

Ветка: `feature/member3-docker`  
Зависимости: Data Engineer (`app.py`), ML Engineer (модель).

| № | Задача | Файл |
| --- | --- | --- |
| 3.1 | Создать Dockerfile: multi-stage build | `docker/Dockerfile` |
| 3.2 | Настроить docker-compose: API, MLflow, Prometheus | `docker-compose.yml` |
| 3.3 | Создать скрипт мониторинга ресурсов | `src/infrastructure_monitoring.py` |
| 3.4 | Настроить `.dockerignore` | `.dockerignore` |
| 3.5 | Протестировать сборку и запуск контейнера | Локально |
| 3.6 | Создать скриншоты дашбордов для отчёта | `reports/monitoring/` |

Ожидаемый результат:

- Working Docker образ.
- `docker-compose` запускает все сервисы.
- Скрипт мониторинга CPU/Memory.
- 2+ скриншота дашбордов.

От чего зависят другие:

- MLOps & QA ждёт рабочий Docker для CI/CD пайплайна.
- Все используют образ для деплоя и демонстрации.

### MLOps & QA Engineer

Ветка: `feature/member4-cicd`  
Зависимости: все участники, код готов.

| № | Задача | Файл |
| --- | --- | --- |
| 4.1 | Написать тесты для ETL | `tests/test_data_processing.py` |
| 4.2 | Написать тесты для модели | `tests/test_model.py` |
| 4.3 | Написать тесты для API | `tests/test_api.py` |
| 4.4 | Настроить GitHub Actions CI/CD | `.github/workflows/ci-cd.yml` |
| 4.5 | Создать презентацию 5-7 слайдов | `presentation/` |
| 4.6 | Настроить `.gitignore` | `.gitignore` |
| 4.7 | Создать релиз `v1.0.0` с тегами | Git tags |

Ожидаемый результат:

- Покрытие тестов > 70%.
- CI/CD пайплайн работает, jobs зелёные.
- Презентация в PDF.
- Репозиторий открыт для преподавателя.

От чего зависят другие:

- Все зависят от рабочего CI/CD для автоматических проверок.

## Критические точки синхронизации

| Точка | Участники | Что синхронизировать |
| --- | --- | --- |
| 1 | Data Engineer -> ML Engineer | Формат обработанных данных `.pkl` |
| 2 | ML Engineer -> DevOps Engineer | Путь к модели в Docker образе |
| 3 | Data Engineer, ML Engineer, DevOps Engineer -> MLOps & QA | Все файлы в `main` для тестирования |
| 4 | Все | Финальный merge в `main` перед сдачей |

## Что должно получиться в итоге

| Артефакт | Где | Кто |
| --- | --- | --- |
| Рабочий ETL пайплайн | `src/data_processing.py` | Data Engineer |
| FastAPI приложение | `src/app.py` | Data Engineer |
| Обученная модель R2 > 0.95 | `models/best_model.pkl` | ML Engineer |
| Метрики модели | `models/metrics.pkl` | ML Engineer |
| Визуализации | `reports/figures/` | ML Engineer |
| Docker образ | `docker/Dockerfile` | DevOps Engineer |
| Docker Compose | `docker-compose.yml` | DevOps Engineer |
| Мониторинг инфраструктуры | `src/infrastructure_monitoring.py` | DevOps Engineer |
| Тесты | `tests/` | MLOps & QA |
| CI/CD пайплайн | `.github/workflows/` | MLOps & QA |
| Презентация 5-7 слайдов | `presentation/` | MLOps & QA |
| README-отчёт | `README.md` | Все |

## Docker

Контейнеризация реализована через:

- `docker/Dockerfile`
- `docker-compose.yml`
- `.dockerignore`

Dockerfile использует:

- multi-stage build;
- production-образ на `python:3.9-slim`;
- non-root пользователя;
- health check;
- переменные окружения для запуска API.

Запуск:

```bash
docker-compose up --build
```

## CI/CD

GitHub Actions workflow находится в `.github/workflows/ci-cd.yml`.

Pipeline включает:

- запуск тестов;
- проверку покрытия;
- сборку Docker-образа;
- линтинг;
- симуляцию деплоя;
- уведомления о результате.

## Мониторинг

Мониторинг качества модели реализован в `src/monitoring.py`:

- проверка дрейфа данных;
- контроль качества предсказаний;
- сравнение текущих метрик с baseline;
- график изменения RMSE во времени.

Мониторинг инфраструктуры реализован в `src/infrastructure_monitoring.py`:

- CPU;
- память;
- диск;
- Docker-контейнеры;
- ресурсные алерты.

## Тестирование

Тесты находятся в папке `tests`:

- `tests/test_data_processing.py`
- `tests/test_model.py`
- `tests/test_api.py`

Запуск:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

Текущая особенность: тесты, требующие обученную модель и препроцессор, пропускаются до появления файлов `models/best_model.pkl` и `models/preprocessor.pkl`.

## Запуск проекта

### Локальный запуск

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/data_processing.py
python src/model_training.py
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

### Запуск через Docker

```bash
docker-compose up --build
```

## Git-flow

Каждый участник работает в отдельной ветке:

- `feature/member1-etl`
- `feature/member2-model`
- `feature/member3-docker`
- `feature/member4-cicd`

Пример команд:

```bash
git checkout -b feature/my-task
git add .
git commit -m "feat: description by [Name]"
git push origin feature/my-task
```

## Финальный чеклист

- [ ] Описание проекта.
- [ ] Описание бизнес-задачи.
- [ ] Описание датасета и признаков.
- [ ] Схема ML-пайплайна.
- [ ] Описание ETL.
- [ ] AutoML или кастомная модель.
- [ ] Метрики модели.
- [ ] Визуализации и графики.
- [ ] Тестирование через `pytest`.
- [ ] Dockerfile и `docker-compose.yml`.
- [ ] CI/CD GitHub Actions.
- [ ] Мониторинг качества модели.
- [ ] Мониторинг инфраструктуры.
- [ ] GitHub-репозиторий и ссылка в отчёте.
- [ ] Доступ преподавателю `@ElenaSmyslovskikh`.
- [ ] Презентация на 5-7 слайдов.

## Дедлайны

- Репозиторий: 25 мая 2026.
- Финальная сдача проекта: 30 мая 2026.
- Защита: последний вебинар.
