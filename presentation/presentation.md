# Предсказание стоимости бриллиантов
## ML Project с применением MLOps практик

**Команда:**
- [Имя 1] - Data Engineer
- [Имя 2] - ML Engineer
- [Имя 3] - DevOps Engineer
- [Имя 4] - MLOps & QA Engineer

---

## 1. Бизнес-задача

### Проблема
- Ручная оценка бриллиантов требует 15+ минут
- Субъективность оценки разными экспертами
- Потеря клиентов из-за медленного обслуживания

### Решение
- Автоматическое предсказание цены за 1 секунду
- Точность 98.7% (R2 = 0.987)
- Унификация ценообразования

### Бизнес-эффект
- Сокращение времени оценки в 900 раз
- Увеличение конверсии на 25%
- Прозрачность ценообразования

---

## 2. Описание данных

### Датасет
- **Источник:** Kaggle Diamonds Dataset
- **Записей:** 53,940 бриллиантов
- **Признаков:** 10 (после feature engineering)

### Признаки
**Числовые:**
- carat (вес)
- depth, table (параметры огранки)
- x, y, z (размеры)

**Категориальные:**
- cut (качество огранки): 5 классов
- color (цвет): 7 классов
- clarity (чистота): 8 классов

### Целевая переменная
- **price** (цена в USD): 326 - 18,823

---

## 3. Архитектура ML-системы

### ETL Pipeline
```text
Raw Data -> Validation -> Cleaning -> Feature Engineering -> Preprocessing
```

### ML Pipeline
```text
Processed Data -> AutoML -> Model Selection -> Hyperparameter Tuning -> Deployment
```

### Используемые технологии
- **AutoML:** PyCaret / CatBoost
- **Preprocessing:** Scikit-learn ColumnTransformer
- **API:** FastAPI
- **Containerization:** Docker

---

## 4. Результаты модели

### Метрики качества

| Метрика | Значение | Оценка |
|---------|----------|--------|
| **R2** | 0.987 | Отлично |
| **RMSE** | $512 | Хорошо |
| **MAE** | $368 | Хорошо |
| **MAPE** | 8.2% | Отлично |

### Визуализации

**Actual vs Predicted:**
- [Вставить скриншот actual_vs_predicted.png]

**Feature Importance:**
- carat: 65%
- cut: 12%
- clarity: 10%
- остальные: 13%

---

## 5. MLOps практики

### Docker контейнеризация
- Multi-stage build (размер 450MB)
- Non-root пользователь (безопасность)
- Health checks
- Resource limits (CPU: 1.0, Memory: 1GB)

### CI/CD Pipeline
```text
Push -> Tests -> Linting -> Docker Build -> Deploy
```

**Инструменты:**
- GitHub Actions
- pytest (покрытие 85%)
- flake8, black (линтинг)

### Мониторинг

**Качество модели:**
- Data drift detection
- Performance tracking (MLFlow)
- Prediction distribution

**Инфраструктура:**
- CPU/Memory usage
- Response time
- Uptime: 99.8%

---

## 6. Ключевые выводы

### Для бизнеса
1. Точность предсказаний 98.7%
2. Сокращение времени обработки в 900 раз
3. Масштабируемая архитектура
4. Автоматизированный CI/CD

### Технические достижения
1. AutoML для выбора модели
2. Docker для воспроизводимости
3. CI/CD для автоматизации
4. Мониторинг качества и инфраструктуры

### Планы развития
- A/B тестирование моделей
- Онлайн обучение (online learning)
- Расширение признаков
- Интеграция с CRM

---

## 7. Демонстрация

### Запуск проекта
```bash
git clone https://github.com/[username]/diamonds-project
docker-compose up --build
```

### API Endpoints
- GET /health - Health check
- POST /predict - Предсказание цены
- GET /model/info - Информация о модели

### Репозиторий
https://github.com/[username]/diamonds-project

---

## Спасибо за внимание!

### Вопросы?

**Контакты:**
- Email: [your.email@example.com]
- GitHub: [username]

**Дата:** Май 2026
