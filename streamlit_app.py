"""
ULTIMATE Advanced Streamlit Dashboard for Crop Yield Prediction System.
Enterprise-grade with interactive visualizations, SHAP explanations, and advanced analytics.
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
from datetime import datetime, timedelta
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
# Advanced CSS Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 0% 0%, #1a2332 0%, #0d1117 100%);
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 20px rgba(102, 187, 106, 0.1); }
        50% { box-shadow: 0 0 40px rgba(102, 187, 106, 0.2); }
    }
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .hero {
        animation: fadeInUp 0.8s ease-out;
        padding: 2.5rem 3rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #1a3a2a 0%, #0d1a1a 100%);
        border: 1px solid #2a4a3a;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle, rgba(102, 187, 106, 0.05) 0%, transparent 70%);
        animation: pulseGlow 3s ease-in-out infinite;
    }
    .hero h1 {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #81C784 0%, #A5D6A7 50%, #81C784 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientFlow 4s ease-in-out infinite;
        margin: 0 0 0.5rem 0;
        position: relative;
        z-index: 1;
    }
    .hero p {
        color: #a0b0a0;
        font-size: 1.1rem;
        position: relative;
        z-index: 1;
    }
    
    .metric-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid #2a3a3a;
        border-radius: 16px;
        padding: 1.5rem;
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
    }
    .metric-card:hover {
        border-color: #4a7a5a;
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    }
    
    .prediction-result {
        background: linear-gradient(135deg, rgba(102, 187, 106, 0.08) 0%, rgba(17, 24, 39, 0.8) 100%);
        border: 1px solid rgba(102, 187, 106, 0.2);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        animation: fadeInUp 0.6s ease-out;
    }
    .prediction-value {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #81C784 0%, #A5D6A7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    
    .shap-card {
        background: rgba(255,255,255,0.02);
        border: 1px solid #2a3a3a;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .badge-advanced {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        background: linear-gradient(135deg, #FF6B6B, #FF8E53);
        color: white;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-left: 0.5rem;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6a7a6a;
        font-size: 0.9rem;
        border-top: 1px solid #1a2a1a;
        margin-top: 3rem;
    }
    
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.02);
        border: 1px solid #2a3a3a;
        border-radius: 12px;
        padding: 1rem;
    }
    div[data-testid="stMetricLabel"] {
        color: #8a9a8a;
    }
    div[data-testid="stMetricValue"] {
        color: #A5D6A7;
    }
</style>
""", unsafe_allow_html=True)

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
if 'shap_values' not in st.session_state:
    st.session_state.shap_values = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None

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
def load_reference_data():
    try:
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "crop_yield_dataset.csv")
        if os.path.exists(data_path):
            return pd.read_csv(data_path)
    except:
        pass
    return None

def make_prediction(payload):
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def get_explanation(payload):
    try:
        response = requests.post(f"{API_URL}/explain", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
    except:
        pass
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
    except:
        pass
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

# ---------------------------------------------------------------------------
# Hero Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🌾 Ultimate Crop Yield Prediction System</h1>
    <p>
        Enterprise-grade AI-powered crop yield prediction with SHAP explanations, 
        confidence intervals, sensitivity analysis, and advanced analytics.
    </p>
    <div style="margin-top: 1rem; display: flex; gap: 1rem; flex-wrap: wrap; position: relative; z-index: 1;">
        <span class="badge-advanced">🚀 AI-Powered</span>
        <span class="badge-advanced" style="background: linear-gradient(135deg, #4CAF50, #45a049);">📊 SHAP Explained</span>
        <span class="badge-advanced" style="background: linear-gradient(135deg, #2196F3, #1976D2);">🎯 95% Confidence</span>
        <span class="badge-advanced" style="background: linear-gradient(135deg, #FF9800, #F57C00);">⚡ Real-time</span>
    </div>
</div>
""", unsafe_allow_html=True)

# API Status in expander
with st.expander("🔌 API Connection Status", expanded=False):
    if st.session_state.api_healthy:
        st.success(f"✅ Connected to: {API_URL}")
        if st.session_state.metadata:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model", st.session_state.metadata.get('model_name', 'XGBoost'))
            with col2:
                st.metric("SHAP Available", "✅" if st.session_state.metadata.get('shap_available') else "❌")
            with col3:
                st.metric("Status", "🟢 Healthy")
    else:
        st.error("❌ API Not Connected")
        if st.button("🔄 Retry Connection"):
            st.cache_data.clear()
            st.rerun()

# Load reference data
ref = load_reference_data()

# ---------------------------------------------------------------------------
# Sidebar - Advanced Navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🎯 Navigation")
    
    selected = option_menu(
        menu_title=None,
        options=[
            "🎯 Predict",
            "🧠 Explain", 
            "📊 Analyze",
            "📈 Forecast",
            "⚡ Compare",
            "📋 History"
        ],
        icons=[
            "target",
            "brain",
            "bar-chart",
            "graph-up",
            "arrows-expand",
            "clock-history"
        ],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#A5D6A7", "font-size": "1.1rem"},
            "nav-link": {
                "font-size": "0.95rem",
                "text-align": "left",
                "margin": "0.2rem 0",
                "border-radius": "10px",
                "padding": "0.6rem 1rem",
                "color": "#a0b0a0",
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg, rgba(102, 187, 106, 0.2), rgba(17, 24, 39, 0.5))",
                "border": "1px solid rgba(102, 187, 106, 0.3)",
                "color": "#A5D6A7",
            },
        }
    )

# ---------------------------------------------------------------------------
# Main Content - Based on Navigation
# ---------------------------------------------------------------------------

# ========== 1. PREDICT TAB ==========
if selected == "🎯 Predict":
    st.markdown("## 🎯 Make a Prediction")
    
    # Get options
    options = st.session_state.options
    if options is None:
        options = fetch_options()
    
    if options is None:
        st.error("Failed to load options. Please check API connection.")
        st.stop()
    
    # Input Form
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("### 📋 Input Parameters")
        
        with st.container():
            countries = options.get('countries', [])
            default_idx = countries.index("India") if "India" in countries else 0
            country = st.selectbox("🌍 Country", countries, index=default_idx)
            
            crops = options.get('crops', [])
            crop = st.selectbox("🌱 Crop", crops)
            
            year_min = options.get('year_min', 1990)
            year_max = options.get('year_max', 2100)
            year = st.number_input("📅 Year", min_value=year_min, max_value=year_max, value=year_max)
        
        st.markdown("### 🌤️ Weather Conditions")
        
        with st.container():
            # Get default values from reference data
            default_rain = 1000.0
            default_temp = 22.0
            default_pest = 10000.0
            
            if ref is not None and country in ref['country'].values:
                hist = ref[ref["country"] == country]
                default_rain = float(hist["rainfall_mm"].mean()) if len(hist) else 1000.0
                default_temp = float(hist["avg_temp_c"].mean()) if len(hist) else 22.0
                default_pest = float(hist["pesticides_tonnes"].mean()) if len(hist) else 10000.0
            
            rainfall = st.slider(
                "💧 Rainfall (mm)",
                0, 3000, int(default_rain),
                help="Average annual rainfall"
            )
            
            temperature = st.slider(
                "🌡️ Temperature (°C)",
                -5, 45, int(round(default_temp)),
                help="Average temperature"
            )
            
            pesticides = st.number_input(
                "🧪 Pesticides (tonnes)",
                min_value=0.0,
                value=float(round(default_pest, 1)),
                step=100.0,
                help="Total pesticide use"
            )
        
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
                    st.markdown("""
                    <div class="prediction-result">
                        <div style="color: #8a9a8a; font-size: 0.9rem;">PREDICTED YIELD</div>
                        <div class="prediction-value">{:.3f} t/ha</div>
                        <div style="color: #8a9a8a; font-size: 0.85rem;">{:.0f} hg/ha</div>
                    </div>
                    """.format(
                        result['predicted_yield_tonnes_per_ha'],
                        result['predicted_yield_hg_per_ha']
                    ), unsafe_allow_html=True)
                    
                    # Metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Model", result['model_name'])
                    with col_b:
                        st.metric("R² Score", f"{result['model_test_r2']:.3f}")
                    with col_c:
                        st.metric("RMSE", f"{result['model_test_rmse_hg_ha']:.1f}")
                    
                    # Confidence interval
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
                    
                    # Store SHAP values for explanation tab
                    if result.get('shap_values') and result.get('feature_names'):
                        st.session_state.shap_values = {
                            'values': result['shap_values'],
                            'features': result['feature_names'],
                            'base_value': result.get('base_value', 0)
                        }
                        st.success("🧠 SHAP explanations available in the 'Explain' tab!")
                    
                    # Quick sensitivity preview
                    if st.button("🔬 Quick Sensitivity Analysis"):
                        sensitivity = get_sensitivity(payload, "rainfall_mm")
                        if sensitivity:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=sensitivity['values'],
                                y=sensitivity['predictions'],
                                mode='lines+markers',
                                line=dict(color='#66BB6A', width=3),
                                marker=dict(size=8, color='#A5D6A7')
                            ))
                            fig.update_layout(
                                template="plotly_dark",
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                title="Sensitivity to Rainfall",
                                xaxis_title="Rainfall (mm)",
                                yaxis_title="Predicted Yield (t/ha)",
                                height=300,
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("❌ Prediction failed. Please try again.")
        else:
            st.info("👈 Fill in the parameters and click 'Predict Yield'")

# ========== 2. EXPLAIN TAB ==========
elif selected == "🧠 Explain":
    st.markdown("## 🧠 SHAP Explanation")
    st.markdown("*Understand what drives each prediction*")
    
    if st.session_state.shap_values is None:
        st.warning("⚠️ No SHAP explanations available. Make a prediction first in the 'Predict' tab.")
        
        # Quick prediction form for explanation
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
                
                if st.button("🔮 Predict & Explain"):
                    payload = {
                        "country": country,
                        "crop": crop,
                        "year": year,
                        "rainfall_mm": rainfall,
                        "avg_temp_c": temp,
                        "pesticides_tonnes": 10000
                    }
                    
                    explanation = get_explanation(payload)
                    if explanation:
                        st.session_state.shap_values = {
                            'values': [f['shap_value'] for f in explanation['feature_contributions']],
                            'features': [f['feature'] for f in explanation['feature_contributions']],
                            'base_value': explanation['base_value']
                        }
                        st.rerun()
    else:
        shap_data = st.session_state.shap_values
        features = shap_data['features']
        values = shap_data['values']
        base_value = shap_data['base_value']
        
        # Summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Base Value", f"{base_value:.2f} hg/ha")
        with col2:
            total_contribution = sum(values)
            st.metric("Total Contribution", f"{total_contribution:.2f}")
        with col3:
            final_pred = base_value + total_contribution
            st.metric("Final Prediction", f"{max(0, final_pred):.2f}")
        
        # Feature contributions
        st.markdown("### 📊 Feature Contributions")
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Feature': features,
            'SHAP Value': values,
            'Impact': ['Positive' if v > 0 else 'Negative' for v in values]
        })
        df = df.sort_values('SHAP Value', ascending=True)
        
        # Horizontal bar chart
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
        
        # Summary of impact
        st.markdown("### 📝 Explanation Summary")
        
        positive = [f for f, v in zip(features, values) if v > 0]
        negative = [f for f, v in zip(features, values) if v < 0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**✅ Positive Contributors (Increased Yield)**")
            for f in sorted(positive, key=lambda x: values[features.index(x)], reverse=True)[:3]:
                st.markdown(f"- {f}: +{values[features.index(f)]:.1f} hg/ha")
            if not positive:
                st.markdown("- None")
        
        with col2:
            st.markdown("**❌ Negative Contributors (Decreased Yield)**")
            for f in sorted(negative, key=lambda x: values[features.index(x)])[:3]:
                st.markdown(f"- {f}: {values[features.index(f)]:.1f} hg/ha")
            if not negative:
                st.markdown("- None")
        
        # Clear button
        if st.button("🗑️ Clear Explanation"):
            st.session_state.shap_values = None
            st.rerun()

# ========== 3. ANALYZE TAB ==========
elif selected == "📊 Analyze":
    st.markdown("## 📊 Advanced Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📈 Sensitivity Analysis", "🏆 Feature Importance", "📊 Model Performance"])
    
    with tab1:
        st.markdown("### 🔬 Sensitivity Analysis")
        st.markdown("*How does changing a parameter affect the prediction?*")
        
        if ref is not None:
            col1, col2 = st.columns(2)
            with col1:
                country = st.selectbox("Country", ref['country'].unique(), key="sens_country")
                crop = st.selectbox("Crop", ref['crop'].unique(), key="sens_crop")
            with col2:
                parameter = st.selectbox(
                    "Parameter to vary",
                    ["rainfall_mm", "avg_temp_c", "pesticides_tonnes", "year"]
                )
            
            if st.button("📊 Analyze Sensitivity"):
                # Get average values for the country
                hist = ref[ref["country"] == country]
                
                payload = {
                    "country": country,
                    "crop": crop,
                    "year": int(hist["year"].mean()) if len(hist) else 2013,
                    "rainfall_mm": float(hist["rainfall_mm"].mean()) if len(hist) else 1000,
                    "avg_temp_c": float(hist["avg_temp_c"].mean()) if len(hist) else 22,
                    "pesticides_tonnes": float(hist["pesticides_tonnes"].mean()) if len(hist) else 10000,
                }
                
                sensitivity = get_sensitivity(payload, parameter)
                
                if sensitivity:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Correlation", f"{sensitivity['correlation']:.3f}")
                    with col2:
                        st.metric("Sensitivity Score", f"{sensitivity['sensitivity_score']:.3f}")
                    with col3:
                        st.metric("Range", f"{sensitivity['max_prediction'] - sensitivity['min_prediction']:.3f}")
                    
                    # Plot
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
    
    with tab2:
        st.markdown("### 🏆 Global Feature Importance")
        
        importance = fetch_feature_importance()
        if importance:
            features = importance['features']
            
            # Create DataFrame
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
            for i, (feature, importance) in enumerate(list(features.items())[:5], 1):
                st.progress(importance, text=f"{i}. {feature}: {importance:.3f}")
        else:
            st.info("Feature importance not available. Train the model first.")
    
    with tab3:
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
            
            # Simulate performance visualization
            st.markdown("### Simulated Performance Metrics")
            
            # Create sample performance chart
            fig = make_subplots(rows=1, cols=2,
                               subplot_titles=("Prediction Error Distribution", "Actual vs Predicted"))
            
            # Error distribution
            errors = np.random.normal(0, 200, 1000)
            fig.add_trace(go.Histogram(x=errors, nbinsx=40, marker_color='#66BB6A'), row=1, col=1)
            
            # Actual vs Predicted
            actual = np.random.uniform(1000, 8000, 100)
            predicted = actual + np.random.normal(0, 200, 100)
            fig.add_trace(go.Scatter(x=actual, y=predicted, mode='markers',
                                    marker=dict(color='#A5D6A7', size=6)), row=1, col=2)
            fig.add_trace(go.Scatter(x=[0, 10000], y=[0, 10000],
                                    mode='lines', line=dict(color='red', dash='dash')), row=1, col=2)
            
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=350,
                showlegend=False,
                font=dict(color="#a0b0a0"),
            )
            fig.update_xaxes(title_text="Error (hg/ha)", row=1, col=1)
            fig.update_xaxes(title_text="Actual (hg/ha)", row=1, col=2)
            fig.update_yaxes(title_text="Count", row=1, col=1)
            fig.update_yaxes(title_text="Predicted (hg/ha)", row=1, col=2)
            
            st.plotly_chart(fig, use_container_width=True)

# ========== 4. FORECAST TAB ==========
elif selected == "📈 Forecast":
    st.markdown("## 📈 Time Series Forecasting")
    st.markdown("*Predict future crop yields based on historical trends*")
    
    if ref is not None:
        col1, col2 = st.columns(2)
        with col1:
            country = st.selectbox("Country", ref['country'].unique(), key="fore_country")
            crop = st.selectbox("Crop", ref['crop'].unique(), key="fore_crop")
        with col2:
            years_ahead = st.slider("Years to forecast", 1, 10, 5)
        
        if st.button("📈 Generate Forecast"):
            try:
                response = requests.post(
                    f"{API_URL}/forecast",
                    params={"country": country, "crop": crop, "years_ahead": years_ahead},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Historical + Forecast
                    fig = go.Figure()
                    
                    # Historical
                    fig.add_trace(go.Scatter(
                        x=data['historical']['years'],
                        y=data['historical']['yields'],
                        mode='lines+markers',
                        name='Historical',
                        line=dict(color='#66BB6A', width=2),
                        marker=dict(size=8, color='#A5D6A7')
                    ))
                    
                    # Forecast
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
                    
                    # Trend info
                    st.markdown("### 📊 Trend Analysis")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Direction", data['trend']['direction'].title())
                    with col2:
                        st.metric("Change per Year", f"{data['trend']['change_per_year']:.1f} hg/ha")
                    with col3:
                        st.metric("Years Forecasted", len(data['forecast']['years']))
                    
                    # Forecast table
                    st.markdown("### 📋 Forecast Table")
                    forecast_df = pd.DataFrame({
                        'Year': data['forecast']['years'],
                        'Predicted Yield (hg/ha)': [round(y, 1) for y in data['forecast']['predicted_yield']]
                    })
                    st.dataframe(forecast_df, use_container_width=True)
                else:
                    st.error("Failed to generate forecast")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Reference data not available for forecasting")

# ========== 5. COMPARE TAB ==========
elif selected == "⚡ Compare":
    st.markdown("## ⚡ Scenario Comparison")
    st.markdown("*Compare multiple scenarios side by side*")
    
    st.info("🔧 Create multiple scenarios to compare their predicted yields.")
    
    # Scenario inputs
    num_scenarios = st.number_input("Number of scenarios", 2, 5, 2)
    
    scenarios = []
    for i in range(num_scenarios):
        st.markdown(f"### Scenario {i+1}")
        col1, col2 = st.columns(2)
        with col1:
            country = st.selectbox(f"Country {i+1}", 
                                  st.session_state.options.get('countries', ['India']), 
                                  key=f"comp_country_{i}")
            crop = st.selectbox(f"Crop {i+1}", 
                               st.session_state.options.get('crops', ['Wheat']), 
                               key=f"comp_crop_{i}")
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
    
    if st.button("⚡ Compare All"):
        try:
            response = requests.post(f"{API_URL}/what-if", json=scenarios, timeout=30)
            if response.status_code == 200:
                data = response.json()
                
                if 'scenarios' in data:
                    # Create comparison chart
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
                        
                        # Best and worst
                        if data.get('best_scenario'):
                            st.success(f"🏆 Best: {data['best_scenario']['scenario']} - {data['best_scenario']['prediction']:.3f} t/ha")
                        if data.get('worst_scenario'):
                            st.warning(f"📉 Worst: {data['worst_scenario']['scenario']} - {data['worst_scenario']['prediction']:.3f} t/ha")
                        
                        # Comparison stats
                        if data.get('comparison'):
                            comp = data['comparison']
                            st.markdown("### 📊 Comparison Statistics")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Range", f"{comp['range']:.3f}")
                            with col2:
                                st.metric("Percent Change", f"{comp['percent_change']:.1f}%")
                            with col3:
                                st.metric("Best Yield", f"{comp['max']:.3f} t/ha")
            else:
                st.error("Comparison failed")
        except Exception as e:
            st.error(f"Error: {e}")

# ========== 6. HISTORY TAB ==========
elif selected == "📋 History":
    st.markdown("## 📋 Prediction History")
    
    if not st.session_state.prediction_history:
        st.info("No predictions made yet. Make a prediction in the 'Predict' tab.")
    else:
        # Create history dataframe
        history_df = pd.DataFrame(st.session_state.prediction_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        
        # Display stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", len(history_df))
        with col2:
            st.metric("Avg Yield", f"{history_df['prediction'].mean():.3f} t/ha")
        with col3:
            st.metric("Max Yield", f"{history_df['prediction'].max():.3f} t/ha")
        
        # History table
        st.markdown("### 📊 Prediction History")
        display_df = history_df[['timestamp', 'country', 'crop', 'year', 'prediction']].copy()
        display_df.columns = ['Time', 'Country', 'Crop', 'Year', 'Yield (t/ha)']
        st.dataframe(display_df, use_container_width=True)
        
        # Historical trend
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
        
        # Clear history
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
    <span style="font-size: 0.8rem; color: #4a5a4a;">
        Built with ❤️ for sustainable agriculture
    </span>
</div>
""", unsafe_allow_html=True)