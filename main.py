"""
FastAPI inference service for the Crop Yield Prediction System.

Run from the project root:
    cd api
    uvicorn main:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paths & model loading - FIXED FOR RAILWAY
# ---------------------------------------------------------------------------
# Get the absolute path to the project root
# If running from root: BASE_DIR = current directory
# If running from api/: BASE_DIR = parent directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# If the models folder doesn't exist in parent, try current directory (for Railway)
MODELS_DIR = os.path.join(BASE_DIR, "models")
if not os.path.exists(MODELS_DIR):
    # Try current directory (if main.py is in root)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "crop_yield_model.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.pkl")

# Global variables
model = None
metadata = None

# ---------------------------------------------------------------------------
# Lifespan context manager (replaces @app.on_event("startup"))
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, metadata
    
    print(f"📁 Current working directory: {os.getcwd()}")
    print(f"📁 Base directory: {BASE_DIR}")
    print(f"📁 Models directory: {MODELS_DIR}")
    print(f"📁 Looking for model at: {MODEL_PATH}")
    
    # Check if models directory exists
    if not os.path.exists(MODELS_DIR):
        print(f"❌ Models directory not found at: {MODELS_DIR}")
        # List what's in the current directory for debugging
        try:
            print(f"📁 Contents of {BASE_DIR}: {os.listdir(BASE_DIR)}")
        except:
            pass
        yield
        return
    
    # List files in models directory
    try:
        model_files = os.listdir(MODELS_DIR)
        print(f"📁 Files in models directory: {model_files}")
    except Exception as e:
        print(f"❌ Cannot list models directory: {e}")
        yield
        return
    
    # Load model
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model file not found at {MODEL_PATH}")
            print(f"⚠️ API will run in limited mode (model not loaded)")
            yield
            return
            
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully from: {MODEL_PATH}")
        
        # Load metadata
        if os.path.exists(METADATA_PATH):
            metadata = joblib.load(METADATA_PATH)
            print(f"✅ Metadata loaded successfully")
            print(f"📊 Model: {metadata.get('model_name', 'Unknown')}")
            print(f"📊 Test R²: {metadata.get('test_r2', 'N/A')}")
        else:
            print(f"⚠️ Metadata file not found at {METADATA_PATH}")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        # Don't raise - allow API to start for debugging
    
    yield  # App runs here
    
    # Shutdown
    print("🛑 Shutting down API")

# ---------------------------------------------------------------------------
# FastAPI app initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Crop Yield Prediction API",
    description="Predict expected crop yield (hg/hectare) from country, crop, and weather data.",
    version="2.0.0",
    lifespan=lifespan,  # Use lifespan instead of startup event
)

# ---------------------------------------------------------------------------
# CORS Middleware - ENHANCED FOR STREAMLIT CLOUD
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # Allow all origins (for development)
        "https://share.streamlit.io",
        "https://*.streamlit.app",
        "https://*.railway.app",
        "https://*.up.railway.app",
        "http://localhost:8501",  # Local Streamlit
        "http://localhost:8000",   # Local FastAPI
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "Accept",
        "Accept-Encoding",
        "Authorization",
        "Content-Type",
        "Origin",
        "User-Agent",
        "X-Requested-With",
    ],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

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
    metadata_loaded: bool = False

class OptionsResponse(BaseModel):
    countries: list[str]
    crops: list[str]
    year_min: int
    year_max: int

class DebugResponse(BaseModel):
    base_dir: str
    models_dir: str
    models_dir_exists: bool
    model_file_exists: bool
    metadata_file_exists: bool
    model_loaded: bool
    metadata_loaded: bool
    current_directory: str
    directory_contents: list[str]
    model_files: list[str]

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
        "debug": "GET /debug",
        "cors_test": "GET /cors-test",
    }

@app.get("/cors-test", tags=["General"])
def cors_test():
    """Endpoint to test CORS configuration"""
    return {
        "status": "success",
        "message": "CORS is working correctly!",
        "headers": dict(os.environ),
    }

@app.get("/debug", response_model=DebugResponse, tags=["General"])
def debug():
    """Debug endpoint to check file structure and model status"""
    directory_contents = []
    if os.path.exists(BASE_DIR):
        try:
            directory_contents = os.listdir(BASE_DIR)
        except:
            pass
    
    model_files = []
    if os.path.exists(MODELS_DIR):
        try:
            model_files = os.listdir(MODELS_DIR)
        except:
            pass
    
    return DebugResponse(
        base_dir=BASE_DIR,
        models_dir=MODELS_DIR,
        models_dir_exists=os.path.exists(MODELS_DIR),
        model_file_exists=os.path.exists(MODEL_PATH),
        metadata_file_exists=os.path.exists(METADATA_PATH),
        model_loaded=model is not None,
        metadata_loaded=metadata is not None,
        current_directory=os.getcwd(),
        directory_contents=directory_contents,
        model_files=model_files,
    )

@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    return HealthResponse(
        status="ok" if model is not None else "model not loaded",
        model_loaded=model is not None,
        model_name=metadata.get("model_name") if metadata else None,
        metadata_loaded=metadata is not None,
    )

@app.get("/options", response_model=OptionsResponse, tags=["General"])
def options():
    """Valid country/crop values the trained model recognizes, plus training year range."""
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return OptionsResponse(
        countries=metadata.get("countries", []),
        crops=metadata.get("crops", []),
        year_min=metadata.get("year_min", 1990),
        year_max=metadata.get("year_max", 2100),
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(payload: YieldInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if metadata is None:
        raise HTTPException(status_code=503, detail="Metadata not loaded")

    # Validate country
    countries = metadata.get("countries", [])
    if countries and payload.country not in countries:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown country '{payload.country}'. See GET /options for valid values.",
        )
    
    # Validate crop
    crops = metadata.get("crops", [])
    if crops and payload.crop not in crops:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown crop '{payload.crop}'. See GET /options for valid values.",
        )

    # Prepare features
    row = pd.DataFrame([payload.model_dump()])
    row_fe = engineer_features(row)

    # Get feature columns
    numeric_features = metadata.get("numeric_features", [])
    categorical_features = metadata.get("categorical_features", [])
    feature_cols = numeric_features + categorical_features
    
    if not feature_cols:
        # Fallback: use all columns
        feature_cols = row_fe.columns.tolist()

    try:
        pred_hg_ha = float(model.predict(row_fe[feature_cols])[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    pred_hg_ha = max(pred_hg_ha, 0.0)
    
    return PredictionResponse(
        predicted_yield_hg_per_ha=round(pred_hg_ha, 1),
        predicted_yield_tonnes_per_ha=round(pred_hg_ha / 10_000, 4),
        model_name=metadata.get("model_name", "Unknown"),
        model_test_r2=round(metadata.get("test_r2", 0.0), 4),
        model_test_rmse_hg_ha=round(metadata.get("test_rmse", 0.0), 1),
    )

@app.options("/{full_path:path}")
async def options_route(full_path: str):
    """Handle preflight OPTIONS requests for all routes"""
    return {
        "message": "OK"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)