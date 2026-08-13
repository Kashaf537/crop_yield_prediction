"""
FastAPI inference service for the Crop Yield Prediction System.

Run from the project root:
    cd api
    uvicorn main:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paths & model loading
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_yield_model.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.pkl")

app = FastAPI(
    title="Crop Yield Prediction API",
    description="Predict expected crop yield (hg/hectare) from country, crop, and weather data.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
metadata = None


@app.on_event("startup")
def load_model():
    global model, metadata
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. "
            "Run the training notebook (notebooks/Crop_Yield_Prediction.ipynb) first."
        )
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    print(f"✅ Loaded model: {metadata['model_name']} "
          f"(test R²={metadata['test_r2']:.3f}, RMSE={metadata['test_rmse']:.0f} hg/ha)")


# ---------------------------------------------------------------------------
# Feature engineering (must mirror src/prepare_dataset.py / the notebook)
# ---------------------------------------------------------------------------
def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["temp_rain_interaction"] = data["avg_temp_c"] * data["rainfall_mm"] / 1000
    data["log_pesticides"] = np.log1p(data["pesticides_tonnes"])
    data["years_since_1990"] = data["year"] - 1990
    return data


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class YieldInput(BaseModel):
    country: str = Field(..., description="Country name, e.g. 'India' (see /options for valid values)")
    crop: str = Field(..., description="Crop name, e.g. 'Wheat' (see /options for valid values)")
    year: int = Field(2013, ge=1990, le=2100, description="Year")
    rainfall_mm: float = Field(..., ge=0, le=5000, description="Average annual rainfall (mm)")
    avg_temp_c: float = Field(..., ge=-10, le=45, description="Average temperature (°C)")
    pesticides_tonnes: float = Field(..., ge=0, le=1_000_000, description="Pesticide use (tonnes)")

    class Config:
        json_schema_extra = {
            "example": {
                "country": "India",
                "crop": "Wheat",
                "year": 2013,
                "rainfall_mm": 1083.0,
                "avg_temp_c": 24.5,
                "pesticides_tonnes": 45000,
            }
        }


class PredictionResponse(BaseModel):
    predicted_yield_hg_per_ha: float
    predicted_yield_tonnes_per_ha: float
    model_name: str
    model_test_r2: float
    model_test_rmse_hg_ha: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None


class OptionsResponse(BaseModel):
    countries: list[str]
    crops: list[str]
    year_min: int
    year_max: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["General"])
def root():
    return {
        "message": "Crop Yield Prediction API",
        "docs": "/docs",
        "health": "/health",
        "options": "GET /options",
        "predict": "POST /predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    return HealthResponse(
        status="ok" if model is not None else "model not loaded",
        model_loaded=model is not None,
        model_name=metadata["model_name"] if metadata else None,
    )


@app.get("/options", response_model=OptionsResponse, tags=["General"])
def options():
    """Valid country/crop values the trained model recognizes, plus training year range."""
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return OptionsResponse(
        countries=metadata["countries"],
        crops=metadata["crops"],
        year_min=metadata["year_min"],
        year_max=metadata["year_max"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(payload: YieldInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if payload.country not in metadata["countries"]:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown country '{payload.country}'. See GET /options for valid values.",
        )
    if payload.crop not in metadata["crops"]:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown crop '{payload.crop}'. See GET /options for valid values.",
        )

    row = pd.DataFrame([payload.model_dump()])
    row_fe = engineer_features(row)

    feature_cols = metadata["numeric_features"] + metadata["categorical_features"]
    try:
        pred_hg_ha = float(model.predict(row_fe[feature_cols])[0])
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    pred_hg_ha = max(pred_hg_ha, 0.0)
    return PredictionResponse(
        predicted_yield_hg_per_ha=round(pred_hg_ha, 1),
        predicted_yield_tonnes_per_ha=round(pred_hg_ha / 10_000, 4),
        model_name=metadata["model_name"],
        model_test_r2=round(metadata["test_r2"], 4),
        model_test_rmse_hg_ha=round(metadata["test_rmse"], 1),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
