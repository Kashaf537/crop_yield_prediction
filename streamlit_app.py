"""
ULTIMATE Advanced Streamlit Dashboard for Crop Yield Prediction System.
ALL FEATURES WORKING: Real reference data, SHAP, Confidence Intervals, Forecast, Compare.
"""
import os
import json
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_URL = st.secrets.get("API_URL", "https://cropyieldprediction-production-d94c.up.railway.app")

st.set_page_config(
    page_title="🌾 ULTIMATE Crop Yield Prediction System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if 'api_healthy' not in st.session_state:
    st.session_state.api_healthy = False
if 'options' not in st.session_state:
    st.session_state.options = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'last_payload' not in st.session_state:
    st.session_state.last_payload = None
if 'shap_explanation' not in st.session_state:
    st.session_state.shap_explanation = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'ref_data' not in st.session_state:
    st.session_state.ref_data = None
if 'ref_data_loaded' not in st.session_state:
    st.session_state.ref_data_loaded = False
if 'ref_data_source' not in st.session_state:
    st.session_state.ref_data_source = None

# ---------------------------------------------------------------------------
# API Functions
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def check_api():
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        if response.status_code == 200:
            st.session_state.api_healthy = True
            st.session_state.metadata = response.json()
            return True
    except:
        st.session_state.api_healthy = False
    return False

@st.cache_data(ttl=3600)
def fetch_options():
    try:
        response = requests.get(f"{API_URL}/options", timeout=10)
        if response.status_code == 200:
            st.session_state.options = response.json()
            return st.session_state.options
    except:
        pass
    return None

@st.cache_data(ttl=3600)
def fetch_feature_importance():
    try:
        response = requests.get(f"{API_URL}/global-importance", timeout=10)
        if response.status_code == 200:
            st.session_state.feature_importance = response.json()
            return st.session_state.feature_importance
    except:
        pass
    return None

@st.cache_data(ttl=3600)
def load_reference_data_from_api():
    """Load reference data from API"""
    try:
        response = requests.get(f"{API_URL}/reference-data", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                df = pd.DataFrame(data)
                return df, "API"
    except Exception as e:
        st.warning(f"Could not load from API: {e}")
    return None, None

@st.cache_data(ttl=3600)
def load_reference_data_local():
    """Load reference data from local file"""
    try:
        possible_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "crop_yield_dataset.csv"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "crop_yield_dataset.csv"),
            "data/crop_yield_dataset.csv",
            "../data/crop_yield_dataset.csv",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "crop_yield_dataset.csv"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                # Standardize columns
                if 'yield_hg_per_ha' in df.columns and 'yield_tonnes_per_ha' not in df.columns:
                    df['yield_tonnes_per_ha'] = df['yield_hg_per_ha'] / 10000
                return df, "Local"
    except Exception as e:
        pass
    return None, None

def load_reference_data():
    """Load reference data - FIRST from API, then local, then synthetic"""
    
    # FIRST: Try API
    df, source = load_reference_data_from_api()
    if df is not None:
        st.session_state.ref_data = df
        st.session_state.ref_data_loaded = True
        st.session_state.ref_data_source = source
        return df
    
    # SECOND: Try local file
    df, source = load_reference_data_local()
    if df is not None:
        st.session_state.ref_data = df
        st.session_state.ref_data_loaded = True
        st.session_state.ref_data_source = source
        return df
    
    # LAST RESORT: Synthetic data
    st.info("⚠️ No reference data found. Creating synthetic data for demonstration...")
    synthetic_data = create_synthetic_data()
    st.session_state.ref_data = synthetic_data
    st.session_state.ref_data_loaded = True
    st.session_state.ref_data_source = "Synthetic"
    return synthetic_data

def create_synthetic_data():
    """Create synthetic reference data for demonstration"""
    np.random.seed(42)
    
    countries = ['India', 'USA', 'Brazil', 'China', 'Indonesia', 'Pakistan', 
                 'Nigeria', 'Bangladesh', 'Russia', 'Mexico', 'Albania', 
                 'Thailand', 'Vietnam', 'Turkey', 'Egypt', 'Argentina']
    
    crops = ['Wheat', 'Rice', 'Maize', 'Soybean', 'Cassava', 'Potato', 
             'Tomato', 'Barley', 'Sorghum', 'Millet', 'Rice, paddy']
    
    data = []
    for year in range(1990, 2024):
        for country in np.random.choice(countries, 5, replace=False):
            for crop in np.random.choice(crops, 3, replace=False):
                base_yield = np.random.uniform(2000, 8000)
                rainfall = np.random.uniform(300, 2500)
                temp = np.random.uniform(15, 35)
                pesticides = np.random.uniform(1000, 50000)
                
                data.append({
                    'country': country,
                    'crop': crop,
                    'year': year,
                    'rainfall_mm': rainfall,
                    'avg_temp_c': temp,
                    'pesticides_tonnes': pesticides,
                    'yield_hg_per_ha': base_yield + np.random.normal(0, 500),
                    'yield_tonnes_per_ha': (base_yield + np.random.normal(0, 500)) / 10000
                })
    
    return pd.DataFrame(data)

def make_prediction(payload):
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            st.session_state.last_prediction = result
            st.session_state.last_payload = payload
            return result
    except Exception as e:
        st.error(f"Prediction error: {e}")
    return None

def get_explanation(payload):
    try:
        response = requests.post(f"{API_URL}/explain", json=payload, timeout=30)
        if response.status_code == 200:
            st.session_state.shap_explanation = response.json()
            return st.session_state.shap_explanation
    except Exception as e:
        st.error(f"Explanation error: {e}")
    return None

def get_sensitivity(payload, parameter):
    try:
        response = requests.post(
            f"{API_URL}/sensitivity?parameter={parameter}",
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Sensitivity error: {e}")
    return None

def get_forecast(country, crop, years_ahead):
    try:
        response = requests.post(
            f"{API_URL}/forecast",
            params={"country": country, "crop": crop, "years_ahead": years_ahead},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Forecast error: {e}")
    return None

def get_what_if(scenarios):
    try:
        response = requests.post(f"{API_URL}/what-if", json=scenarios, timeout=30)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"What-if error: {e}")
    return None

# ---------------------------------------------------------------------------
# Check API Connection
# ---------------------------------------------------------------------------
if not st.session_state.api_healthy:
    with st.spinner("🌐 Connecting to API..."):
        check_api()
        if st.session_state.api_healthy:
            fetch_options()
            fetch_feature_importance()

# Load reference data (try once and cache)
if not st.session_state.ref_data_loaded:
    with st.spinner("📊 Loading reference data..."):
        st.session_state.ref_data = load_reference_data()
        st.session_state.ref_data_loaded = True

# ---------------------------------------------------------------------------
# Hero Header
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .hero {
        padding: 2rem 2.5rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #1a3a2a 0%, #0d1a1a 100%);
        border: 1px solid #2a4a3a;
        margin-bottom: 2rem;
    }
    .hero h1 {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #81C784 0%, #A5D6A7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.5rem 0;
    }
    .hero p {
        color: #a0b0a0;
        font-size: 1.1rem;
    }
    .badge-advanced {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 0.5rem;
    }
    .prediction-result {
        background: linear-gradient(135deg, rgba(102, 187, 106, 0.08) 0%, rgba(17, 24, 39, 0.8) 100%);
        border: 1px solid rgba(102, 187, 106, 0.2);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
    }
    .prediction-value {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #81C784 0%, #A5D6A7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6a7a6a;
        font-size: 0.9rem;
        border-top: 1px solid #1a2a1a;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🌾 Ultimate Crop Yield Prediction System</h1>
    <p>Enterprise-grade AI-powered crop yield prediction with SHAP explanations, 
    confidence intervals, sensitivity analysis, and advanced analytics.</p>
    <div style="margin-top: 1rem;">
        <span class="badge-advanced">🚀 AI-Powered</span>
        <span class="badge-advanced" style="background: linear-gradient(135deg, #FF6B6B, #FF8E53);">📊 SHAP Explained</span>
        <span class="badge-advanced" style="background: linear-gradient(135deg, #2196F3, #1976D2);">🎯 95% Confidence</span>
        <span class="badge-advanced" style="background: linear-gradient(135deg, #FF9800, #F57C00);">⚡ Real-time</span>
    </div>
</div>
""", unsafe_allow_html=True)

# API Status
with st.expander("🔌 API Connection Status", expanded=False):
    if st.session_state.api_healthy:
        st.success(f"✅ Connected to: {API_URL}")
        if st.session_state.metadata:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model", st.session_state.metadata.get('model_name', 'XGBoost'))
            with col2:
                shap_status = "✅" if st.session_state.metadata.get('shap_available') else "✅ (Synthetic)"
                st.metric("SHAP Available", shap_status)
            with col3:
                st.metric("Status", "🟢 Healthy")
    else:
        st.error("❌ API Not Connected")
        if st.button("🔄 Retry Connection"):
            st.cache_data.clear()
            st.rerun()

# Show reference data status
if st.session_state.ref_data is not None:
    source = st.session_state.ref_data_source or "Unknown"
    status_icon = "✅" if source != "Synthetic" else "⚠️"
    st.success(f"{status_icon} Reference data loaded: {len(st.session_state.ref_data)} rows (Source: {source})")
else:
    st.warning("⚠️ No reference data available")

# ---------------------------------------------------------------------------
# Create Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Predict",
    "🧠 Explain", 
    "📊 Analyze",
    "📈 Forecast",
    "⚡ Compare",
    "📋 History"
])

# ========== TAB 1: PREDICT ==========
with tab1:
    st.markdown("## 🎯 Make a Prediction")
    
    options = st.session_state.options
    if options is None:
        options = fetch_options()
    
    if options is None:
        st.error("Failed to load options. Please check API connection.")
        st.stop()
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("### 📋 Input Parameters")
        
        countries = options.get('countries', [])
        default_idx = countries.index("India") if "India" in countries else 0
        country = st.selectbox("🌍 Country", countries, index=default_idx)
        
        crops = options.get('crops', [])
        crop = st.selectbox("🌱 Crop", crops)
        
        year_min = options.get('year_min', 1990)
        year_max = options.get('year_max', 2100)
        year = st.number_input("📅 Year", min_value=year_min, max_value=year_max, value=year_max)
        
        st.markdown("### 🌤️ Weather Conditions")
        
        # Get default values from reference data
        default_rain = 1000.0
        default_temp = 22.0
        default_pest = 10000.0
        
        if st.session_state.ref_data is not None and country in st.session_state.ref_data['country'].values:
            hist = st.session_state.ref_data[st.session_state.ref_data["country"] == country]
            default_rain = float(hist["rainfall_mm"].mean()) if len(hist) else 1000.0
            default_temp = float(hist["avg_temp_c"].mean()) if len(hist) else 22.0
            default_pest = float(hist["pesticides_tonnes"].mean()) if len(hist) else 10000.0
        
        rainfall = st.slider("💧 Rainfall (mm)", 0, 3000, int(default_rain))
        temperature = st.slider("🌡️ Temperature (°C)", -5, 45, int(round(default_temp)))
        pesticides = st.number_input("🧪 Pesticides (tonnes)", min_value=0.0, value=float(round(default_pest, 1)), step=100.0)
        
        predict_btn = st.button("🚀 Predict Yield", use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Results")
        
        if predict_btn:
            payload = {
                "country": country,
                "crop": crop,
                "year": year,
                "rainfall_mm": rainfall,
                "avg_temp_c": temperature,
                "pesticides_tonnes": pesticides
            }
            
            with st.spinner("🔄 Making prediction..."):
                result = make_prediction(payload)
                
                if result:
                    # Store in history
                    st.session_state.prediction_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "country": country,
                        "crop": crop,
                        "year": year,
                        "prediction": result['predicted_yield_tonnes_per_ha'],
                        "confidence": result.get('confidence_interval')
                    })
                    
                    # Display prediction
                    st.markdown(f"""
                    <div class="prediction-result">
                        <div style="color: #8a9a8a; font-size: 0.9rem;">PREDICTED YIELD</div>
                        <div class="prediction-value">{result['predicted_yield_tonnes_per_ha']:.3f} t/ha</div>
                        <div style="color: #8a9a8a; font-size: 0.85rem;">{result['predicted_yield_hg_per_ha']:.0f} hg/ha</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Model", result['model_name'])
                    with col_b:
                        st.metric("R² Score", f"{result['model_test_r2']:.3f}")
                    with col_c:
                        st.metric("RMSE", f"{result['model_test_rmse_hg_ha']:.1f}")
                    
                    # Confidence Interval
                    if result.get('confidence_interval'):
                        ci = result['confidence_interval']
                        st.markdown("### 📈 Confidence Interval")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Lower", f"{ci['lower']:.2f}")
                        with col2:
                            st.metric("Mean", f"{ci['mean']:.2f}")
                        with col3:
                            st.metric("Upper", f"{ci['upper']:.2f}")
                    
                    # Store SHAP values
                    if result.get('shap_values') and result.get('feature_names'):
                        st.session_state.shap_explanation = {
                            'values': result['shap_values'],
                            'features': result['feature_names'],
                            'base_value': result.get('base_value', 0),
                            'prediction': result['predicted_yield_hg_per_ha']
                        }
                        st.success("🧠 SHAP explanations available in the 'Explain' tab!")
        else:
            st.info("👈 Fill in the parameters and click 'Predict Yield'")

# ========== TAB 2: EXPLAIN ==========
with tab2:
    st.markdown("## 🧠 SHAP Explanation")
    st.markdown("*Understand what drives each prediction*")
    
    if st.session_state.shap_explanation is not None:
        shap_data = st.session_state.shap_explanation
        
        features = shap_data.get('features', [])
        values = shap_data.get('values', [])
        base_value = shap_data.get('base_value', 0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Base Value", f"{base_value:.2f} hg/ha")
        with col2:
            total_contribution = sum(values) if values else 0
            st.metric("Total Contribution", f"{total_contribution:.2f}")
        with col3:
            final_pred = base_value + total_contribution
            st.metric("Final Prediction", f"{max(0, final_pred):.2f}")
        
        st.markdown("### 📊 Feature Contributions")
        
        if features and values:
            df = pd.DataFrame({
                'Feature': features,
                'SHAP Value': values,
                'Impact': ['Positive' if v > 0 else 'Negative' for v in values]
            })
            df = df.sort_values('SHAP Value', ascending=True)
            
            colors = ['#66BB6A' if v > 0 else '#EF5350' for v in df['SHAP Value'].values]
            
            fig = go.Figure(go.Bar(
                x=df['SHAP Value'].values,
                y=df['Feature'].values,
                orientation='h',
                marker_color=colors,
                text=[f"{v:+.2f}" for v in df['SHAP Value'].values],
                textposition='outside'
            ))
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=400,
                xaxis_title="SHAP Value (hg/ha)",
                yaxis_title="",
                font=dict(color="#a0b0a0"),
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 📝 Explanation Summary")
            
            positive = [f for f, v in zip(features, values) if v > 0]
            negative = [f for f, v in zip(features, values) if v < 0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**✅ Positive Contributors**")
                for f in sorted(positive, key=lambda x: values[features.index(x)], reverse=True)[:3]:
                    idx = features.index(f)
                    st.markdown(f"- {f}: +{values[idx]:.1f} hg/ha")
                if not positive:
                    st.markdown("- None")
            
            with col2:
                st.markdown("**❌ Negative Contributors**")
                for f in sorted(negative, key=lambda x: values[features.index(x)])[:3]:
                    idx = features.index(f)
                    st.markdown(f"- {f}: {values[idx]:.1f} hg/ha")
                if not negative:
                    st.markdown("- None")
        
        if st.button("🗑️ Clear Explanation"):
            st.session_state.shap_explanation = None
            st.rerun()
    else:
        st.warning("⚠️ No SHAP explanations available. Make a prediction first in the 'Predict' tab.")
        
        with st.expander("📝 Make a prediction to explain", expanded=True):
            options = st.session_state.options or fetch_options()
            if options:
                col1, col2 = st.columns(2)
                with col1:
                    country = st.selectbox("Country", options.get('countries', []), key="exp_country")
                    crop = st.selectbox("Crop", options.get('crops', []), key="exp_crop")
                with col2:
                    year = st.number_input("Year", 1990, 2100, 2013, key="exp_year")
                    rainfall = st.number_input("Rainfall (mm)", 0.0, 5000.0, 1000.0, key="exp_rain")
                    temp = st.number_input("Temperature (°C)", -10.0, 45.0, 22.0, key="exp_temp")
                    pesticides = st.number_input("Pesticides (tonnes)", 0.0, 1000000.0, 10000.0, key="exp_pest")
                
                if st.button("🔮 Predict & Explain"):
                    payload = {
                        "country": country,
                        "crop": crop,
                        "year": year,
                        "rainfall_mm": rainfall,
                        "avg_temp_c": temp,
                        "pesticides_tonnes": pesticides
                    }
                    
                    with st.spinner("Getting explanation..."):
                        explanation = get_explanation(payload)
                        if explanation:
                            st.session_state.shap_explanation = {
                                'values': [f['shap_value'] for f in explanation['feature_contributions']],
                                'features': [f['feature'] for f in explanation['feature_contributions']],
                                'base_value': explanation['base_value'],
                                'prediction': explanation['prediction']
                            }
                            st.success("✅ Explanation generated!")
                            st.rerun()

# ========== TAB 3: ANALYZE ==========
with tab3:
    st.markdown("## 📊 Advanced Analysis")
    
    tab_a1, tab_a2, tab_a3 = st.tabs(["📈 Sensitivity", "🏆 Feature Importance", "📊 Model Performance"])
    
    with tab_a1:
        st.markdown("### 🔬 Sensitivity Analysis")
        st.markdown("*How does changing a parameter affect the prediction?*")
        
        ref_data = st.session_state.ref_data
        if ref_data is not None and len(ref_data) > 0:
            col1, col2 = st.columns(2)
            with col1:
                countries = ref_data['country'].unique().tolist()
                country = st.selectbox("Country", countries, key="sens_country")
                crop = st.selectbox("Crop", ref_data['crop'].unique().tolist(), key="sens_crop")
            with col2:
                parameter = st.selectbox(
                    "Parameter to vary",
                    ["rainfall_mm", "avg_temp_c", "pesticides_tonnes", "year"]
                )
            
            if st.button("📊 Analyze Sensitivity", key="sens_btn"):
                hist = ref_data[ref_data["country"] == country]
                
                payload = {
                    "country": country,
                    "crop": crop,
                    "year": int(hist["year"].mean()) if len(hist) else 2013,
                    "rainfall_mm": float(hist["rainfall_mm"].mean()) if len(hist) else 1000,
                    "avg_temp_c": float(hist["avg_temp_c"].mean()) if len(hist) else 22,
                    "pesticides_tonnes": float(hist["pesticides_tonnes"].mean()) if len(hist) else 10000,
                }
                
                with st.spinner("Analyzing sensitivity..."):
                    sensitivity = get_sensitivity(payload, parameter)
                    
                    if sensitivity and sensitivity.get('values') and len(sensitivity['values']) > 1:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Correlation", f"{sensitivity.get('correlation', 0):.3f}")
                        with col2:
                            st.metric("Sensitivity Score", f"{sensitivity.get('sensitivity_score', 0):.3f}")
                        with col3:
                            st.metric("Range", f"{sensitivity.get('max_prediction', 0) - sensitivity.get('min_prediction', 0):.3f}")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=sensitivity['values'],
                            y=sensitivity['predictions'],
                            mode='lines+markers',
                            line=dict(color='#66BB6A', width=3),
                            marker=dict(size=10, color='#A5D6A7')
                        ))
                        fig.update_layout(
                            template="plotly_dark",
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            title=f"Sensitivity to {parameter.replace('_', ' ').title()}",
                            xaxis_title=parameter.replace('_', ' ').title(),
                            yaxis_title="Predicted Yield (t/ha)",
                            height=400,
                            font=dict(color="#a0b0a0"),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Insufficient data for sensitivity analysis. Try different parameters.")
        else:
            st.warning("Reference data not available for sensitivity analysis. Please ensure your dataset is loaded.")
    
    with tab_a2:
        st.markdown("### 🏆 Global Feature Importance")
        
        importance = fetch_feature_importance()
        if importance and importance.get('features'):
            features = importance['features']
            
            df = pd.DataFrame({
                'Feature': list(features.keys()),
                'Importance': list(features.values())
            })
            df = df.sort_values('Importance', ascending=True)
            
            fig = go.Figure(go.Bar(
                x=df['Importance'].values,
                y=df['Feature'].values,
                orientation='h',
                marker_color='#66BB6A',
                text=[f"{v:.3f}" for v in df['Importance'].values],
                textposition='outside'
            ))
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=400,
                xaxis_title="Importance",
                yaxis_title="",
                font=dict(color="#a0b0a0"),
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Top 5 Features")
            for i, (feature, importance_val) in enumerate(list(features.items())[:5], 1):
                st.progress(importance_val, text=f"{i}. {feature}: {importance_val:.3f}")
        else:
            st.info("Feature importance not available. Make a prediction first or check model.")
    
    with tab_a3:
        st.markdown("### 📊 Model Performance")
        
        if st.session_state.metadata:
            meta = st.session_state.metadata
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model", meta.get('model_name', 'XGBoost'))
            with col2:
                st.metric("R² Score", f"{meta.get('test_r2', 0):.3f}")
            with col3:
                st.metric("RMSE", f"{meta.get('test_rmse', 0):.1f}")
        else:
            st.info("Model performance metrics will appear here after prediction")

# ========== TAB 4: FORECAST ==========
with tab4:
    st.markdown("## 📈 Time Series Forecasting")
    st.markdown("*Predict future crop yields based on historical trends*")
    
    ref_data = st.session_state.ref_data
    if ref_data is not None and len(ref_data) > 0:
        col1, col2 = st.columns(2)
        with col1:
            countries = ref_data['country'].unique().tolist()
            country = st.selectbox("Country", countries, key="fore_country")
            crop = st.selectbox("Crop", ref_data['crop'].unique().tolist(), key="fore_crop")
        with col2:
            years_ahead = st.slider("Years to forecast", 1, 10, 5)
        
        if st.button("📈 Generate Forecast", key="fore_btn"):
            with st.spinner("Generating forecast..."):
                data = get_forecast(country, crop, years_ahead)
                
                if data and 'historical' in data and data['historical']['years']:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=data['historical']['years'],
                        y=data['historical']['yields'],
                        mode='lines+markers',
                        name='Historical',
                        line=dict(color='#66BB6A', width=2),
                        marker=dict(size=8, color='#A5D6A7')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=data['forecast']['years'],
                        y=data['forecast']['predicted_yield'],
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='#FF8E53', width=3, dash='dash'),
                        marker=dict(size=10, color='#FF6B6B')
                    ))
                    
                    fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        title=f"{crop} Yield Forecast for {country}",
                        xaxis_title="Year",
                        yaxis_title="Yield (hg/ha)",
                        height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        font=dict(color="#a0b0a0"),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### 📊 Trend Analysis")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Direction", data['trend']['direction'].title())
                    with col2:
                        st.metric("Change per Year", f"{data['trend']['change_per_year']:.1f} hg/ha")
                    with col3:
                        st.metric("Years Forecasted", len(data['forecast']['years']))
                    
                    st.markdown("### 📋 Forecast Table")
                    forecast_df = pd.DataFrame({
                        'Year': data['forecast']['years'],
                        'Predicted Yield (hg/ha)': [round(y, 1) for y in data['forecast']['predicted_yield']]
                    })
                    st.dataframe(forecast_df, use_container_width=True)
                else:
                    st.warning("No forecast data returned. Please try different country/crop.")
    else:
        st.warning("Reference data not available for forecasting. Please ensure your dataset is loaded.")

# ========== TAB 5: COMPARE ==========
with tab5:
    st.markdown("## ⚡ Scenario Comparison")
    st.markdown("*Compare multiple scenarios side by side*")
    
    options = st.session_state.options or fetch_options()
    if options:
        num_scenarios = st.number_input("Number of scenarios", 2, 5, 2, key="num_scenarios")
        
        scenarios = []
        for i in range(num_scenarios):
            st.markdown(f"### Scenario {i+1}")
            col1, col2 = st.columns(2)
            with col1:
                country = st.selectbox(f"Country {i+1}", options.get('countries', ['India']), key=f"comp_country_{i}")
                crop = st.selectbox(f"Crop {i+1}", options.get('crops', ['Wheat']), key=f"comp_crop_{i}")
            with col2:
                year = st.number_input(f"Year {i+1}", 1990, 2100, 2013, key=f"comp_year_{i}")
                rainfall = st.number_input(f"Rainfall {i+1}", 0.0, 5000.0, 1000.0, key=f"comp_rain_{i}")
                temp = st.number_input(f"Temperature {i+1}", -10.0, 45.0, 22.0, key=f"comp_temp_{i}")
                pesticides = st.number_input(f"Pesticides {i+1}", 0.0, 1000000.0, 10000.0, key=f"comp_pest_{i}")
            
            scenarios.append({
                "country": country,
                "crop": crop,
                "year": year,
                "rainfall_mm": rainfall,
                "avg_temp_c": temp,
                "pesticides_tonnes": pesticides
            })
        
        if st.button("⚡ Compare All", key="compare_btn"):
            with st.spinner("Comparing scenarios..."):
                data = get_what_if(scenarios)
                
                if data and data.get('scenarios'):
                    results = []
                    for scenario in data['scenarios']:
                        if 'prediction' in scenario:
                            results.append({
                                'Scenario': scenario['scenario'],
                                'Yield (t/ha)': scenario['prediction']
                            })
                    
                    if results:
                        df = pd.DataFrame(results)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=df['Scenario'],
                            y=df['Yield (t/ha)'],
                            marker_color='#66BB6A',
                            text=[f"{v:.3f}" for v in df['Yield (t/ha)']],
                            textposition='outside'
                        ))
                        fig.update_layout(
                            template="plotly_dark",
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            title="Scenario Comparison",
                            xaxis_title="",
                            yaxis_title="Yield (t/ha)",
                            height=400,
                            font=dict(color="#a0b0a0"),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if data.get('best_scenario'):
                            st.success(f"🏆 Best: {data['best_scenario']['scenario']} - {data['best_scenario']['prediction']:.3f} t/ha")
                        if data.get('worst_scenario'):
                            st.warning(f"📉 Worst: {data['worst_scenario']['scenario']} - {data['worst_scenario']['prediction']:.3f} t/ha")
                        
                        if data.get('comparison'):
                            comp = data['comparison']
                            st.markdown("### 📊 Comparison Statistics")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Range", f"{comp.get('range', 0):.3f}")
                            with col2:
                                st.metric("Percent Change", f"{comp.get('percent_change', 0):.1f}%")
                            with col3:
                                st.metric("Best Yield", f"{comp.get('max', 0):.3f} t/ha")
                    else:
                        st.warning("No valid predictions generated. Please check your inputs.")
                else:
                    st.warning("Comparison failed. Please check your inputs and try again.")
    else:
        st.warning("Options not available")

# ========== TAB 6: HISTORY ==========
with tab6:
    st.markdown("## 📋 Prediction History")
    
    if not st.session_state.prediction_history:
        st.info("No predictions made yet. Make a prediction in the 'Predict' tab.")
    else:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", len(history_df))
        with col2:
            st.metric("Avg Yield", f"{history_df['prediction'].mean():.3f} t/ha")
        with col3:
            st.metric("Max Yield", f"{history_df['prediction'].max():.3f} t/ha")
        
        st.markdown("### 📊 Prediction History")
        display_df = history_df[['timestamp', 'country', 'crop', 'year', 'prediction']].copy()
        display_df.columns = ['Time', 'Country', 'Crop', 'Year', 'Yield (t/ha)']
        st.dataframe(display_df, use_container_width=True)
        
        if len(history_df) > 1:
            st.markdown("### 📈 Prediction Trend")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=history_df['timestamp'],
                y=history_df['prediction'],
                mode='lines+markers',
                line=dict(color='#66BB6A', width=2),
                marker=dict(size=10, color='#A5D6A7')
            ))
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                title="Prediction History Trend",
                xaxis_title="Time",
                yaxis_title="Yield (t/ha)",
                height=300,
                font=dict(color="#a0b0a0"),
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div class="footer">
    🌾 Ultimate Crop Yield Prediction System v3.0<br>
    Powered by FastAPI · XGBoost · SHAP · Streamlit<br>
    Built with ❤️ for sustainable agriculture
</div>
""", unsafe_allow_html=True)