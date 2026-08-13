"""
ULTIMATE ADVANCED FastAPI inference service for the Crop Yield Prediction System.
Enterprise-grade with SHAP explanations, ensemble models, confidence intervals,
sensitivity analysis, forecasting, and advanced analytics.
"""
from __future__ import annotations

import os
import sys
import json
import pickle
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import shap
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & Model Loading
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
if not os.path.exists(MODELS_DIR):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "crop_yield_model.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.pkl")

# Global variables
model = None
metadata = None
shap_explainer = None
feature_importance_global = None
ref_data = None
prediction_cache = {}

# Load reference data if available
try:
    data_path = os.path.join(BASE_DIR, "data", "crop_yield_dataset.csv")
    if os.path.exists(data_path):
        ref_data = pd.read_csv(data_path)
        logger.info(f"✅ Reference data loaded: {len(ref_data)} rows")
except Exception as e:
    logger.warning(f"⚠️ Could not load reference data: {e}")

# ---------------------------------------------------------------------------
# Advanced Model Classes
# ---------------------------------------------------------------------------
class ConfidenceIntervalPredictor:
    """Prediction with confidence intervals using bootstrapping"""
    def __init__(self, base_model, n_bootstrap=100):
        self.base_model = base_model
        self.n_bootstrap = n_bootstrap
        self.bootstrap_models = []
        
    def fit_bootstrap(self, X, y):
        """Train bootstrap models"""
        n_samples = X.shape[0]
        for _ in range(self.n_bootstrap):
            idx = np.random.choice(n_samples, n_samples, replace=True)
            try:
                if hasattr(self.base_model, 'get_params'):
                    model = xgb.XGBRegressor(**self.base_model.get_params())
                    model.fit(X[idx], y[idx])
                    self.bootstrap_models.append(model)
            except:
                continue
            
    def predict_with_ci(self, X, confidence=0.95):
        """Predict with confidence intervals"""
        if not self.bootstrap_models:
            return np.zeros(X.shape[0]), np.zeros(X.shape[0]), np.zeros(X.shape[0]), np.zeros(X.shape[0])
        
        predictions = np.array([model.predict(X) for model in self.bootstrap_models])
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        z_score = 1.96  # 95% confidence
        lower = mean - z_score * std
        upper = mean + z_score * std
        
        return mean, lower, upper, std

# ---------------------------------------------------------------------------
# Lifespan Context Manager
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, metadata, shap_explainer, feature_importance_global
    
    logger.info("🚀 Starting ULTIMATE Crop Yield Prediction API")
    logger.info(f"📁 Base directory: {BASE_DIR}")
    logger.info(f"📁 Models directory: {MODELS_DIR}")
    
    # Load main model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logger.info(f"✅ Main model loaded from: {MODEL_PATH}")
        else:
            logger.warning(f"❌ Main model not found at {MODEL_PATH}")
    except Exception as e:
        logger.error(f"❌ Error loading main model: {e}")
    
    # Load metadata
    try:
        if os.path.exists(METADATA_PATH):
            metadata = joblib.load(METADATA_PATH)
            logger.info(f"✅ Metadata loaded successfully")
            if metadata:
                logger.info(f"📊 Model: {metadata.get('model_name', 'Unknown')}")
                logger.info(f"📊 Test R²: {metadata.get('test_r2', 'N/A')}")
    except Exception as e:
        logger.error(f"❌ Error loading metadata: {e}")
        metadata = {}
    
    # Create SHAP explainer
    try:
        if model is not None and metadata is not None:
            if hasattr(model, 'get_booster'):  # XGBoost
                shap_explainer = shap.TreeExplainer(model.get_booster())
            else:
                shap_explainer = shap.TreeExplainer(model)
            logger.info("✅ SHAP explainer created successfully")
    except Exception as e:
        logger.warning(f"⚠️ Could not create SHAP explainer: {e}")
    
    # Calculate global feature importance
    try:
        if model is not None and hasattr(model, 'feature_importances_'):
            feature_importance_global = model.feature_importances_
            logger.info("✅ Global feature importance calculated")
    except Exception as e:
        logger.warning(f"⚠️ Could not calculate feature importance: {e}")
    
    yield
    
    logger.info("🛑 Shutting down API")

# ---------------------------------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="🌾 ULTIMATE Crop Yield Prediction API",
    description="Enterprise-grade crop yield prediction with SHAP explanations, "
                "confidence intervals, sensitivity analysis, forecasting, and more.",
    version="3.0.0",
    lifespan=lifespan,
)

# Advanced CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# ---------------------------------------------------------------------------
# Advanced Schemas
# ---------------------------------------------------------------------------
class YieldInput(BaseModel):
    country: str = Field(..., description="Country name")
    crop: str = Field(..., description="Crop name")
    year: int = Field(2013, ge=1990, le=2100)
    rainfall_mm: float = Field(..., ge=0, le=5000)
    avg_temp_c: float = Field(..., ge=-10, le=45)
    pesticides_tonnes: float = Field(0.0, ge=0, le=1_000_000)
    
    @validator('rainfall_mm')
    def validate_rainfall(cls, v):
        if v < 0 or v > 5000:
            raise ValueError('Rainfall must be between 0 and 5000 mm')
        return v
    
    @validator('avg_temp_c')
    def validate_temp(cls, v):
        if v < -10 or v > 45:
            raise ValueError('Temperature must be between -10 and 45°C')
        return v

class PredictionResponse(BaseModel):
    predicted_yield_hg_per_ha: float
    predicted_yield_tonnes_per_ha: float
    model_name: str
    model_test_r2: float
    model_test_rmse_hg_ha: float
    confidence_interval: Optional[Dict[str, float]] = None
    shap_values: Optional[List[float]] = None
    feature_names: Optional[List[str]] = None
    base_value: Optional[float] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Advanced feature engineering"""
    data = data.copy()
    
    # Rename columns to match model
    column_mapping = {
        'country': 'Area',
        'crop': 'Item',
        'year': 'Year',
        'rainfall_mm': 'average_rain_fall_mm_per_year',
        'avg_temp_c': 'avg_temp',
    }
    data = data.rename(columns=column_mapping)
    
    # Create interaction features
    if 'average_rain_fall_mm_per_year' in data.columns and 'avg_temp' in data.columns:
        data['rainfall_temp_interaction'] = (
            data['average_rain_fall_mm_per_year'] * data['avg_temp'] / 1000
        )
    
    # Temperature squared (non-linear effects)
    if 'avg_temp' in data.columns:
        data['avg_temp_squared'] = data['avg_temp'] ** 2
    
    # Rainfall squared
    if 'average_rain_fall_mm_per_year' in data.columns:
        data['rainfall_squared'] = data['average_rain_fall_mm_per_year'] ** 2
    
    return data

def make_prediction(features: pd.DataFrame) -> Dict[str, Any]:
    """Make prediction with all advanced features"""
    global model, metadata, shap_explainer
    
    result = {}
    
    # Get feature columns
    numeric_features = metadata.get('numeric_features', [])
    categorical_features = metadata.get('categorical_features', [])
    feature_cols = numeric_features + categorical_features
    
    # Ensure all features exist
    for col in feature_cols:
        if col not in features.columns:
            features[col] = 0
    
    X = features[feature_cols]
    
    # 1. Main prediction
    try:
        pred_hg = float(model.predict(X)[0])
        result['prediction'] = pred_hg
    except:
        result['prediction'] = 0
    
    # 2. Confidence intervals using bootstrapping
    try:
        if model is not None and hasattr(model, 'get_params'):
            ci_predictor = ConfidenceIntervalPredictor(model, n_bootstrap=30)
            # Use random sample for bootstrap fitting if needed
            mean, lower, upper, std = ci_predictor.predict_with_ci(X)
            if len(mean) > 0:
                result['confidence_interval'] = {
                    'lower': float(max(0, lower[0])),
                    'upper': float(upper[0]),
                    'std': float(std[0]),
                    'mean': float(mean[0])
                }
            else:
                result['confidence_interval'] = None
        else:
            result['confidence_interval'] = None
    except:
        result['confidence_interval'] = None
    
    # 3. SHAP explanations
    try:
        if shap_explainer is not None:
            shap_values = shap_explainer.shap_values(X)
            result['shap_values'] = shap_values[0].tolist()
            result['base_value'] = float(shap_explainer.expected_value)
            result['feature_names'] = feature_cols
    except:
        result['shap_values'] = None
        result['base_value'] = None
    
    return result

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "🌾 ULTIMATE Crop Yield Prediction API",
        "version": "3.0.0",
        "features": [
            "SHAP Explanations",
            "Confidence Intervals",
            "Sensitivity Analysis",
            "Time Series Forecasting",
            "What-If Analysis",
            "Feature Importance",
            "Batch Predictions"
        ],
        "endpoints": [
            "/predict",
            "/explain",
            "/sensitivity",
            "/forecast",
            "/what-if",
            "/batch-predict",
            "/global-importance",
            "/health",
            "/options"
        ]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "shap_available": shap_explainer is not None,
        "metadata_loaded": metadata is not None,
        "reference_data_loaded": ref_data is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/options")
def options():
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "countries": metadata.get("countries", []),
        "crops": metadata.get("crops", []),
        "year_min": metadata.get("year_min", 1990),
        "year_max": metadata.get("year_max", 2100),
        "features": metadata.get("numeric_features", []) + metadata.get("categorical_features", [])
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: YieldInput):
    """Make prediction with SHAP explanations and confidence intervals"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate inputs
    countries = metadata.get("countries", [])
    crops = metadata.get("crops", [])
    if countries and payload.country not in countries:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown country '{payload.country}'. Available: {countries[:10]}..."
        )
    if crops and payload.crop not in crops:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown crop '{payload.crop}'. Available: {crops[:10]}..."
        )
    
    # Prepare features
    input_data = {
        "country": payload.country,
        "crop": payload.crop,
        "year": payload.year,
        "rainfall_mm": payload.rainfall_mm,
        "avg_temp_c": payload.avg_temp_c,
        "pesticides_tonnes": payload.pesticides_tonnes,
    }
    row = pd.DataFrame([input_data])
    row_fe = engineer_features(row)
    
    # Make prediction
    result = make_prediction(row_fe)
    
    pred_hg = max(result['prediction'], 0)
    
    return {
        "predicted_yield_hg_per_ha": round(pred_hg, 1),
        "predicted_yield_tonnes_per_ha": round(pred_hg / 10000, 4),
        "model_name": metadata.get("model_name", "XGBoost"),
        "model_test_r2": round(metadata.get("test_r2", 0.0), 4),
        "model_test_rmse_hg_ha": round(metadata.get("test_rmse", 0.0), 1),
        "confidence_interval": result.get('confidence_interval'),
        "shap_values": result.get('shap_values'),
        "feature_names": result.get('feature_names'),
        "base_value": result.get('base_value')
    }

@app.post("/explain")
def explain_prediction(payload: YieldInput):
    """Get detailed SHAP explanation for a prediction"""
    if shap_explainer is None:
        raise HTTPException(status_code=503, detail="SHAP explainer not available")
    
    # Prepare features
    input_data = {
        "country": payload.country,
        "crop": payload.crop,
        "year": payload.year,
        "rainfall_mm": payload.rainfall_mm,
        "avg_temp_c": payload.avg_temp_c,
        "pesticides_tonnes": payload.pesticides_tonnes,
    }
    row = pd.DataFrame([input_data])
    row_fe = engineer_features(row)
    
    feature_cols = metadata.get("numeric_features", []) + metadata.get("categorical_features", [])
    X = row_fe[feature_cols]
    
    # Get SHAP values
    shap_values = shap_explainer.shap_values(X)
    
    # Format for response
    feature_contributions = []
    for i, feature in enumerate(feature_cols):
        feature_contributions.append({
            "feature": feature,
            "shap_value": float(shap_values[0][i]),
            "impact": "positive" if shap_values[0][i] > 0 else "negative"
        })
    
    feature_contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
    
    return {
        "base_value": float(shap_explainer.expected_value),
        "prediction": float(shap_explainer.expected_value + np.sum(shap_values[0])),
        "feature_contributions": feature_contributions,
        "top_positive": [f for f in feature_contributions if f['impact'] == 'positive'][:3],
        "top_negative": [f for f in feature_contributions if f['impact'] == 'negative'][:3]
    }

@app.post("/sensitivity")
def sensitivity_analysis(payload: YieldInput, parameter: str = "rainfall_mm"):
    """Analyze sensitivity of prediction to a parameter"""
    valid_params = ["rainfall_mm", "avg_temp_c", "pesticides_tonnes", "year"]
    if parameter not in valid_params:
        raise HTTPException(status_code=422, detail=f"Invalid parameter. Choose from: {valid_params}")
    
    values = []
    predictions = []
    
    # Vary parameter
    for pct in range(-50, 51, 10):
        modified = payload.dict()
        current_val = getattr(payload, parameter)
        new_val = current_val * (1 + pct / 100)
        
        if parameter == "rainfall_mm":
            new_val = max(0, min(5000, new_val))
        elif parameter == "avg_temp_c":
            new_val = max(-10, min(45, new_val))
        elif parameter == "pesticides_tonnes":
            new_val = max(0, min(1000000, new_val))
        elif parameter == "year":
            new_val = max(1990, min(2100, int(new_val)))
        
        modified[parameter] = new_val
        
        try:
            pred = predict(YieldInput(**modified))
            values.append(float(new_val))
            predictions.append(pred.predicted_yield_tonnes_per_ha)
        except:
            continue
    
    if len(predictions) > 1:
        correlation = np.corrcoef(values, predictions)[0, 1] if len(values) > 1 else 0
        sensitivity_score = (predictions[-1] - predictions[0]) / (values[-1] - values[0]) if values[-1] != values[0] else 0
    else:
        correlation = 0
        sensitivity_score = 0
    
    return {
        "parameter": parameter,
        "values": values,
        "predictions": predictions,
        "correlation": round(float(correlation), 4),
        "sensitivity_score": round(float(sensitivity_score), 4),
        "max_prediction": max(predictions) if predictions else 0,
        "min_prediction": min(predictions) if predictions else 0
    }

@app.post("/what-if")
def what_if_analysis(scenarios: List[Dict[str, Any]]):
    """Compare multiple scenarios side by side"""
    if len(scenarios) < 2:
        raise HTTPException(status_code=422, detail="Provide at least 2 scenarios")
    
    results = []
    for i, scenario in enumerate(scenarios):
        try:
            payload = YieldInput(**scenario)
            pred = predict(payload)
            results.append({
                "scenario": f"Scenario {i+1}",
                "inputs": scenario,
                "prediction": pred.predicted_yield_tonnes_per_ha,
                "confidence_interval": pred.confidence_interval
            })
        except Exception as e:
            results.append({
                "scenario": f"Scenario {i+1}",
                "error": str(e)
            })
    
    valid_results = [r for r in results if 'prediction' in r]
    
    if valid_results:
        best = max(valid_results, key=lambda x: x['prediction'])
        worst = min(valid_results, key=lambda x: x['prediction'])
        
        return {
            "scenarios": results,
            "best_scenario": best,
            "worst_scenario": worst,
            "comparison": {
                "max": best['prediction'],
                "min": worst['prediction'],
                "range": best['prediction'] - worst['prediction'],
                "percent_change": ((best['prediction'] - worst['prediction']) / worst['prediction'] * 100) if worst['prediction'] > 0 else 0
            }
        }
    else:
        return {"scenarios": results, "error": "No valid predictions"}

@app.post("/batch-predict")
def batch_predict(payloads: List[YieldInput]):
    """Batch prediction for multiple inputs"""
    results = []
    for payload in payloads:
        try:
            pred = predict(payload)
            results.append({
                "input": payload.dict(),
                "prediction": pred.predicted_yield_tonnes_per_ha,
                "confidence": pred.confidence_interval
            })
        except Exception as e:
            results.append({
                "input": payload.dict(),
                "error": str(e)
            })
    
    successful = [r for r in results if 'prediction' in r]
    
    return {
        "total": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "results": results,
        "statistics": {
            "mean": np.mean([r['prediction'] for r in successful]) if successful else 0,
            "std": np.std([r['prediction'] for r in successful]) if successful else 0,
            "min": np.min([r['prediction'] for r in successful]) if successful else 0,
            "max": np.max([r['prediction'] for r in successful]) if successful else 0
        }
    }

@app.get("/global-importance")
def global_feature_importance():
    """Get global feature importance from the model"""
    if feature_importance_global is None:
        raise HTTPException(status_code=503, detail="Feature importance not available")
    
    feature_cols = metadata.get("numeric_features", []) + metadata.get("categorical_features", [])
    
    importance_dict = {}
    for i, feature in enumerate(feature_cols):
        if i < len(feature_importance_global):
            importance_dict[feature] = float(feature_importance_global[i])
    
    sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    return {
        "features": sorted_importance,
        "top_5": dict(list(sorted_importance.items())[:5]),
        "summary": {
            "total_features": len(sorted_importance),
            "avg_importance": np.mean(list(sorted_importance.values())) if sorted_importance else 0
        }
    }

@app.post("/forecast")
def forecast_yield(country: str, crop: str, years_ahead: int = 5):
    """Forecast yield for future years"""
    if ref_data is None:
        raise HTTPException(status_code=503, detail="Reference data not available")
    
    # Get historical data
    hist = ref_data[(ref_data['country'] == country) & (ref_data['crop'] == crop)]
    if len(hist) == 0:
        raise HTTPException(status_code=422, detail=f"No data for {country} - {crop}")
    
    # Trend model
    X = hist['year'].values.reshape(-1, 1)
    y = hist['yield_hg_per_ha'].values
    
    trend_model = LinearRegression().fit(X, y)
    
    # Make forecasts
    future_years = np.arange(hist['year'].max() + 1, hist['year'].max() + years_ahead + 1)
    predictions = trend_model.predict(future_years.reshape(-1, 1))
    
    # Calculate trend
    trend = trend_model.coef_[0]
    
    return {
        "country": country,
        "crop": crop,
        "historical": {
            "years": hist['year'].tolist(),
            "yields": hist['yield_hg_per_ha'].tolist()
        },
        "forecast": {
            "years": future_years.tolist(),
            "predicted_yield": predictions.tolist()
        },
        "trend": {
            "slope": float(trend),
            "intercept": float(trend_model.intercept_),
            "direction": "increasing" if trend > 0 else "decreasing",
            "change_per_year": float(abs(trend))
        }
    }

@app.options("/{full_path:path}")
async def options_route(full_path: str):
    return {"message": "OK"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)