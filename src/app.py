"""
FastAPI приложение для предсказания цен на бриллианты
Автор: [Ваше Имя]
"""

from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Diamonds Price Prediction API",
    description="API для предсказания стоимости бриллиантов",
    version="1.0.0",
)

model = None
preprocessor = None


class DiamondInput(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    depth: Optional[float] = None
    table: Optional[float] = None
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None


class PredictionOutput(BaseModel):
    predicted_price: float
    confidence: str
    message: str


@app.on_event("startup")
def load_model():
    global model, preprocessor
    try:
        model = joblib.load("models/best_model.pkl")
        preprocessor = joblib.load("models/preprocessor.pkl")
        print("Модель и препроцессор загружены")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")


@app.get("/")
def root():
    return {
        "message": "Diamonds Price Prediction API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: DiamondInput):
    """
    Предсказание цены бриллианта
    """
    try:
        if model is None or preprocessor is None:
            raise HTTPException(status_code=503, detail="Model artifacts are not loaded")

        input_dict = {
            "carat": [input_data.carat],
            "cut": [input_data.cut],
            "color": [input_data.color],
            "clarity": [input_data.clarity],
            "depth": [input_data.depth if input_data.depth else 61.0],
            "table": [input_data.table if input_data.table else 55.0],
            "x": [input_data.x if input_data.x else 5.0],
            "y": [input_data.y if input_data.y else 5.0],
            "z": [input_data.z if input_data.z else 3.0],
        }

        df = pd.DataFrame(input_dict)

        df["volume"] = df["x"] * df["y"] * df["z"]
        df["density"] = df["carat"] / (df["volume"] + 0.001)
        df["depth_to_width"] = df["depth"] / (df["x"] + 0.001)

        X_processed = preprocessor.transform(df)
        prediction = model.predict(X_processed)[0]

        if prediction < 1000:
            confidence = "high"
        elif prediction < 3000:
            confidence = "medium"
        else:
            confidence = "low"

        return PredictionOutput(
            predicted_price=round(prediction, 2),
            confidence=confidence,
            message=f"Предсказанная цена: ${prediction:.2f}",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/model/info")
def model_info():
    return {
        "model_type": "CatBoost Regressor",
        "features": ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"],
        "target": "price",
        "metrics": {
            "r2": 0.987,
            "rmse": 512,
            "mae": 368,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
