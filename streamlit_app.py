"""
Streamlit dashboard for the Crop Yield Prediction System.

This app connects to the FastAPI backend deployed on Railway.
"""
import os
import time
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration - Get API URL from Streamlit Cloud secrets
# ---------------------------------------------------------------------------
API_URL = st.secrets.get("API_URL", "https://cropyieldprediction-production-d94c.up.railway.app")

st.set_page_config(
    page_title="Crop Yield Prediction System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Theme / custom CSS
# ---------------------------------------------------------------------------
PRIMARY = "#2E7D32"       # deep green
PRIMARY_LIGHT = "#66BB6A"
ACCENT = "#F9A825"        # warm amber accent
BG_CARD = "#111827"
BORDER = "#1F2937"

st.markdown(f"""
<style>
    .stApp {{
        background: radial-gradient(circle at 15% 0%, #16221a 0%, #0e1117 45%);
    }}
    #MainMenu, footer, header {{ visibility: hidden; }}

    .hero {{
        padding: 2.2rem 2.4rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #14301c 0%, #0e1117 100%);
        border: 1px solid {BORDER};
        margin-bottom: 1.6rem;
    }}
    .hero h1 {{
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0 0 0.35rem 0;
        color: #F1F8E9;
        letter-spacing: -0.02em;
    }}
    .hero p {{
        color: #A7B0A0;
        font-size: 1.02rem;
        margin: 0;
        max-width: 780px;
    }}
    .badge-row {{ margin-top: 1rem; display: flex; gap: 0.6rem; flex-wrap: wrap; }}
    .badge {{
        display: inline-block;
        padding: 0.3rem 0.85rem;
        border-radius: 999px;
        background: rgba(102, 187, 106, 0.12);
        border: 1px solid rgba(102, 187, 106, 0.35);
        color: {PRIMARY_LIGHT};
        font-size: 0.82rem;
        font-weight: 600;
    }}

    .section-card {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.4rem;
    }}
    .section-title {{
        font-size: 1.15rem;
        font-weight: 700;
        color: #F1F8E9;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    .section-sub {{
        color: #8B9389;
        font-size: 0.88rem;
        margin-bottom: 1.1rem;
    }}

    div[data-testid="stMetric"] {{
        background: rgba(255,255,255,0.02);
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 0.85rem 1rem 0.6rem 1rem;
    }}
    div[data-testid="stMetricLabel"] {{ color: #8B9389; }}

    div.stButton > button {{
        background: linear-gradient(120deg, {PRIMARY} 0%, {PRIMARY_LIGHT} 100%);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        font-size: 1rem;
        transition: transform 0.06s ease-in-out;
    }}
    div.stButton > button:hover {{ transform: translateY(-1px); opacity: 0.95; }}

    .result-hero {{
        background: linear-gradient(120deg, rgba(46,125,50,0.18) 0%, rgba(17,24,39,0.4) 100%);
        border: 1px solid rgba(102, 187, 106, 0.35);
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        text-align: center;
        margin-bottom: 1rem;
    }}
    .result-value {{
        font-size: 2.6rem;
        font-weight: 800;
        color: {PRIMARY_LIGHT};
        margin: 0.2rem 0 0.1rem 0;
    }}
    .result-label {{ color: #A7B0A0; font-size: 0.95rem; }}
    .result-sub {{ color: #8B9389; font-size: 0.85rem; margin-top: 0.4rem; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {BG_CARD};
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.1rem;
        color: #8B9389;
    }}
    .stTabs [aria-selected="true"] {{ color: {PRIMARY_LIGHT} !important; }}

    footer-note {{ color: #6B7280; font-size: 0.8rem; }}
    
    .api-status {{
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    .api-status.success {{
        background: rgba(102, 187, 106, 0.15);
        border: 1px solid rgba(102, 187, 106, 0.3);
        color: {PRIMARY_LIGHT};
    }}
    .api-status.error {{
        background: rgba(239, 83, 80, 0.15);
        border: 1px solid rgba(239, 83, 80, 0.3);
        color: #EF5350;
    }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if 'api_healthy' not in st.session_state:
    st.session_state.api_healthy = False
if 'api_checked' not in st.session_state:
    st.session_state.api_checked = False
if 'options' not in st.session_state:
    st.session_state.options = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = None

# ---------------------------------------------------------------------------
# API Connection Functions
# ---------------------------------------------------------------------------
def check_api_connection():
    """Check if the API is reachable and healthy."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        if response.status_code == 200:
            st.session_state.api_healthy = True
            st.session_state.metadata = response.json()
            return True
        else:
            st.session_state.api_healthy = False
            return False
    except requests.exceptions.ConnectionError:
        st.session_state.api_healthy = False
        return False
    except requests.exceptions.Timeout:
        st.session_state.api_healthy = False
        return False
    except Exception:
        st.session_state.api_healthy = False
        return False

@st.cache_data(ttl=3600)
def fetch_options():
    """Fetch available countries, crops, and year range from the API."""
    try:
        response = requests.get(f"{API_URL}/options", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception:
        return None

@st.cache_data(ttl=3600)
def fetch_cors_test():
    """Test CORS configuration."""
    try:
        response = requests.get(f"{API_URL}/cors-test", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception:
        return None

@st.cache_data(ttl=3600)
def load_reference_data():
    """Load the reference dataset from local file."""
    try:
        # Try multiple possible paths
        possible_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "crop_yield_dataset.csv"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "crop_yield_dataset.csv"),
            "data/crop_yield_dataset.csv",
            "../data/crop_yield_dataset.csv",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return pd.read_csv(path)
        
        return None
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Check API health
# ---------------------------------------------------------------------------
if not st.session_state.api_checked:
    with st.spinner("Checking API connection..."):
        st.session_state.api_checked = True
        check_api_connection()
        if st.session_state.api_healthy:
            st.session_state.options = fetch_options()

# Display API status
if st.session_state.api_healthy:
    st.markdown(
        f'<div class="api-status success">✅ Connected to API: {API_URL}</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f'<div class="api-status error">❌ Cannot connect to API. Please check if the API is running.</div>',
        unsafe_allow_html=True
    )
    st.info(f"API URL: {API_URL}")
    
    # Add retry button
    if st.button("🔄 Retry Connection"):
        st.session_state.api_checked = False
        st.rerun()
    st.stop()

# Get options from API
options = st.session_state.options
if options is None:
    options = fetch_options()
    if options is None:
        st.error("Failed to load options from API.")
        st.stop()

# Get metadata
api_metadata = st.session_state.metadata

# Load reference data (for visualizations)
ref = load_reference_data()

# Get model info from API
model_name = api_metadata.get("model_name", "XGBoost") if api_metadata else "XGBoost"

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
    <h1>🌾 Crop Yield Prediction System</h1>
    <p>Estimate expected crop yield from country, crop type, and weather conditions using a
    gradient-boosted regression model trained on two decades of global agricultural data.</p>
    <div class="badge-row">
        <span class="badge">{model_name}</span>
        <span class="badge">{len(options.get('countries', []))} countries</span>
        <span class="badge">{len(options.get('crops', []))} crops</span>
        <span class="badge">{options.get('year_min', 1990)}–{options.get('year_max', 2100)}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input form — front and center, always visible
# ---------------------------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">📋 Prediction Inputs</div>'
    '<div class="section-sub">Select a country and crop, then adjust weather conditions to '
    'generate a yield estimate.</div>',
    unsafe_allow_html=True,
)

with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        countries = options.get('countries', [])
        default_country_idx = countries.index("India") if "India" in countries else 0
        country = st.selectbox("Country", countries, index=default_country_idx)
    with c2:
        crops = options.get('crops', [])
        crop = st.selectbox("Crop", crops)
    with c3:
        year_min = options.get('year_min', 1990)
        year_max = options.get('year_max', 2100)
        year = st.number_input(
            "Year", min_value=year_min, max_value=year_max,
            value=year_max, step=1,
            help=f"Model trained on {year_min}–{year_max} data only."
        )

    # Pre-fill with averages if reference data available
    default_rain = 1000.0
    default_temp = 22.0
    default_pest = 10000.0
    
    if ref is not None and country in ref['country'].values:
        hist = ref[ref["country"] == country]
        default_rain = float(hist["rainfall_mm"].mean()) if len(hist) else 1000.0
        default_temp = float(hist["avg_temp_c"].mean()) if len(hist) else 22.0
        default_pest = float(hist["pesticides_tonnes"].mean()) if len(hist) else 10000.0

    st.markdown("&nbsp;", unsafe_allow_html=True)
    w1, w2, w3 = st.columns(3)
    with w1:
        rainfall_mm = st.slider(
            "Avg. Annual Rainfall (mm)", 0, 3000, int(default_rain),
            help="Pre-filled with this country's historical average"
        )
    with w2:
        avg_temp_c = st.slider(
            "Avg. Temperature (°C)", -5, 45, int(round(default_temp)),
            help="Pre-filled with this country's historical average"
        )
    with w3:
        pesticides_tonnes = st.number_input(
            "Pesticide Use (tonnes)", min_value=0.0, value=round(default_pest, 1), step=100.0,
            help="Total pesticide use for this country"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔮  Predict Yield", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if submitted:
    payload = {
        "country": country,
        "crop": crop,
        "year": year,
        "rainfall_mm": rainfall_mm,
        "avg_temp_c": avg_temp_c,
        "pesticides_tonnes": pesticides_tonnes,
    }
    
    with st.spinner("Predicting..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                pred_hg_ha = result['predicted_yield_hg_per_ha']
                pred_tonnes_ha = result['predicted_yield_tonnes_per_ha']
                
                # Calculate vs global average if reference data available
                crop_avg = None
                delta_pct = 0
                arrow = "▲"
                
                if ref is not None:
                    crop_data = ref[ref["crop"] == crop]
                    if len(crop_data) > 0:
                        crop_avg = crop_data["yield_hg_per_ha"].mean()
                        delta_pct = (pred_hg_ha - crop_avg) / crop_avg * 100 if crop_avg else 0
                        arrow = "▲" if delta_pct >= 0 else "▼"
                
                # Display prediction result
                st.markdown(f"""
                <div class="result-hero">
                    <div class="result-label">PREDICTED YIELD — {crop.upper()} · {country.upper()} · {year}</div>
                    <div class="result-value">{pred_tonnes_ha:.3f} t/ha</div>
                    <div class="result-sub">{pred_hg_ha:,.0f} hg/ha</div>
                </div>
                """, unsafe_allow_html=True)
                
                if crop_avg:
                    st.caption(f"{arrow} {abs(delta_pct):.1f}% vs. global average for {crop} ({crop_avg/10000:.3f} t/ha)")
                
                # Display model info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model", result.get('model_name', 'XGBoost'))
                with col2:
                    st.metric("Test R² Score", f"{result.get('model_test_r2', 0):.3f}")
                with col3:
                    st.metric("Test RMSE", f"{result.get('model_test_rmse_hg_ha', 0):,.0f} hg/ha")
                
                # Visualizations if reference data available
                if ref is not None:
                    st.markdown('<div class="section-card">', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="section-title">📊 {crop} Yield Distribution</div>'
                        f'<div class="section-sub">Where this prediction falls relative to historical '
                        f'{crop} yields across all countries.</div>',
                        unsafe_allow_html=True,
                    )
                    crop_data = ref[ref["crop"] == crop]
                    if len(crop_data) > 0:
                        fig = px.histogram(crop_data, x="yield_tonnes_per_ha", nbins=40)
                        fig.add_vline(x=pred_tonnes_ha, line_color=ACCENT, line_width=3,
                                       annotation_text="Your prediction", annotation_position="top",
                                       annotation_font_color=ACCENT)
                        fig.update_layout(
                            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=10, r=10, t=10, b=10), height=320,
                            xaxis_title="Yield (tonnes/ha)", yaxis_title="Count",
                            bargap=0.05,
                        )
                        fig.update_traces(marker_color=PRIMARY_LIGHT, marker_line_width=0)
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    st.markdown('</div>', unsafe_allow_html=True)
                
            elif response.status_code == 422:
                st.error("Invalid input. Please check your parameters.")
                try:
                    st.json(response.json())
                except:
                    st.write(response.text)
            else:
                st.error(f"API Error: {response.status_code}")
                try:
                    st.json(response.json())
                except:
                    st.write(response.text)
                    
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Please check if the API is running.")
            # Try to reconnect
            if st.button("🔄 Reconnect to API"):
                st.session_state.api_checked = False
                st.rerun()
        except requests.exceptions.Timeout:
            st.error("❌ Request timed out. Please try again.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            
else:
    st.info("Fill in the form above and click **Predict Yield** to generate an estimate.", icon="🌱")

# ---------------------------------------------------------------------------
# Dataset explorer (only if reference data available)
# ---------------------------------------------------------------------------
if ref is not None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">🌍 Explore the Dataset</div>'
        '<div class="section-sub">Historical yield patterns across crops, countries, and time.</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["Yield by Crop", "Top Countries", "Trend Over Time"])

    plotly_layout = dict(
        template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10), height=380,
    )

    with tab1:
        order = ref.groupby("crop")["yield_tonnes_per_ha"].median().sort_values().index.tolist()
        fig1 = px.box(ref, x="crop", y="yield_tonnes_per_ha", color="crop",
                      category_orders={"crop": order})
        fig1.update_layout(**plotly_layout, showlegend=False,
                            xaxis_title="", yaxis_title="Yield (tonnes/ha)")
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

    with tab2:
        top = ref.groupby("country")["yield_tonnes_per_ha"].median().sort_values(ascending=False).head(15)
        fig2 = go.Figure(go.Bar(
            x=top.values, y=top.index, orientation="h",
            marker_color=PRIMARY_LIGHT,
        ))
        fig2.update_layout(**plotly_layout, yaxis=dict(autorange="reversed"),
                            xaxis_title="Median Yield (tonnes/ha)", yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with tab3:
        yearly = ref.groupby("year")["yield_tonnes_per_ha"].mean().reset_index()
        fig3 = px.line(yearly, x="year", y="yield_tonnes_per_ha", markers=True)
        fig3.update_traces(line_color=ACCENT, marker=dict(size=6, color=PRIMARY_LIGHT))
        fig3.update_layout(**plotly_layout, xaxis_title="Year", yaxis_title="Avg Yield (tonnes/ha)")
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("🔎  Preview raw dataset"):
        st.dataframe(ref.head(50), use_container_width=True)

# ---------------------------------------------------------------------------
# Debug section in sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔧 Debug Info")
    st.write(f"**API URL:** {API_URL}")
    st.write(f"**API Status:** {'✅ Connected' if st.session_state.api_healthy else '❌ Disconnected'}")
    st.write(f"**Model:** {model_name}")
    
    if ref is not None:
        st.write(f"**Reference Data:** {len(ref)} rows")
    
    # CORS Test
    if st.button("🧪 Test CORS"):
        with st.spinner("Testing CORS..."):
            cors_result = fetch_cors_test()
            if cors_result:
                st.success("✅ CORS Test Passed!")
                st.json(cors_result)
            else:
                st.error("❌ CORS Test Failed")

st.markdown(
    f'<p class="footer-note">Model served via FastAPI · API: {API_URL}</p>',
    unsafe_allow_html=True,
)