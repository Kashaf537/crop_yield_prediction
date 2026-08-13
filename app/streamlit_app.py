"""
Streamlit dashboard for the Crop Yield Prediction System.

Run from the project root:
    cd app
    streamlit run streamlit_app.py

Self-contained: loads the trained pipeline directly from ../models/crop_yield_model.pkl.
"""
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_yield_model.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "crop_yield_dataset.csv")

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
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data / model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH), joblib.load(METADATA_PATH)


@st.cache_data
def load_reference_data():
    return pd.read_csv(DATA_PATH)


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["temp_rain_interaction"] = data["avg_temp_c"] * data["rainfall_mm"] / 1000
    data["log_pesticides"] = np.log1p(data["pesticides_tonnes"])
    data["years_since_1990"] = data["year"] - 1990
    return data


FEATURE_LABELS = {
    "rainfall_mm": "Rainfall",
    "avg_temp_c": "Temperature",
    "pesticides_tonnes": "Pesticide Use",
    "log_pesticides": "Pesticide Use (log)",
    "temp_rain_interaction": "Temp × Rainfall",
    "year": "Year",
    "years_since_1990": "Years Since 1990",
}


def explain_prediction(pipeline, metadata, input_fe: pd.DataFrame, feature_cols: list):
    """
    Returns a pandas Series of {readable feature name: signed contribution to this
    single prediction, in hg/ha}, aggregated so one-hot country/crop columns collapse
    back into a single 'Country' / 'Crop Type' entry. Uses XGBoost's native SHAP-exact
    pred_contribs. Returns None if the underlying model isn't XGBoost (falls back to
    global feature importance in the UI instead).
    """
    xgb_model = pipeline.named_steps["model"]
    if not isinstance(xgb_model, xgb.XGBRegressor):
        return None

    pre = pipeline.named_steps["preprocess"]
    transformed = pre.transform(input_fe[feature_cols])
    cat_names = pre.named_transformers_["cat"].get_feature_names_out(metadata["categorical_features"])
    all_names = metadata["numeric_features"] + list(cat_names)

    booster = xgb_model.get_booster()
    dmat = xgb.DMatrix(transformed, feature_names=all_names)
    contribs = booster.predict(dmat, pred_contribs=True)[0][:-1]  # drop bias term

    grouped = {}
    for name, val in zip(all_names, contribs):
        if name.startswith("country_"):
            grouped["Country"] = grouped.get("Country", 0.0) + float(val)
        elif name.startswith("crop_"):
            grouped["Crop Type"] = grouped.get("Crop Type", 0.0) + float(val)
        else:
            grouped[FEATURE_LABELS.get(name, name)] = float(val)

    return pd.Series(grouped).sort_values(key=abs, ascending=False)


model, metadata = load_model()
ref = load_reference_data()

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
    <h1>🌾 Crop Yield Prediction System</h1>
    <p>Estimate expected crop yield from country, crop type, and weather conditions using a
    gradient-boosted regression model trained on two decades of global agricultural data.</p>
    <div class="badge-row">
        <span class="badge">XGBoost · R² {metadata['test_r2']:.3f}</span>
        <span class="badge">101 countries</span>
        <span class="badge">10 crops</span>
        <span class="badge">{metadata['year_min']}–{metadata['year_max']}</span>
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
        default_country_idx = (
            metadata["countries"].index("India") if "India" in metadata["countries"] else 0
        )
        country = st.selectbox("Country", metadata["countries"], index=default_country_idx)
    with c2:
        crop = st.selectbox("Crop", metadata["crops"])
    with c3:
        year = st.number_input(
            "Year", min_value=metadata["year_min"], max_value=metadata["year_max"],
            value=metadata["year_max"], step=1,
            help=f"Model trained on {metadata['year_min']}–{metadata['year_max']} data only; "
                 f"capped at {metadata['year_max']} to avoid unreliable extrapolation."
        )

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
    submitted = st.form_submit_button("🔮  Predict Yield", width='stretch')

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if submitted:
    input_df = pd.DataFrame([{
        "country": country, "crop": crop, "year": year,
        "rainfall_mm": rainfall_mm, "avg_temp_c": avg_temp_c,
        "pesticides_tonnes": pesticides_tonnes,
    }])
    input_fe = engineer_features(input_df)
    feature_cols = metadata["numeric_features"] + metadata["categorical_features"]
    pred_hg_ha = max(float(model.predict(input_fe[feature_cols])[0]), 0.0)
    pred_tonnes_ha = pred_hg_ha / 10_000

    crop_avg = ref.loc[ref["crop"] == crop, "yield_hg_per_ha"].mean()
    delta_pct = (pred_hg_ha - crop_avg) / crop_avg * 100 if crop_avg else 0
    arrow = "▲" if delta_pct >= 0 else "▼"

    st.markdown(f"""
    <div class="result-hero">
        <div class="result-label">PREDICTED YIELD — {crop.upper()} · {country.upper()} · {year}</div>
        <div class="result-value">{pred_tonnes_ha:.3f} t/ha</div>
        <div class="result-sub">{pred_hg_ha:,.0f} hg/ha &nbsp;·&nbsp;
            {arrow} {abs(delta_pct):.1f}% vs. global average for {crop} ({crop_avg/10000:.3f} t/ha)</div>
    </div>
    """, unsafe_allow_html=True)

    chart_col, meta_col = st.columns([2, 1])

    with chart_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="section-title">📊 {crop} Yield Distribution</div>'
            f'<div class="section-sub">Where this prediction falls relative to historical '
            f'{crop} yields across all countries.</div>',
            unsafe_allow_html=True,
        )
        crop_data = ref[ref["crop"] == crop]
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
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with meta_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧾 Model Details</div>', unsafe_allow_html=True)
        st.metric("Algorithm", metadata["model_name"])
        st.metric("Test R² Score", f"{metadata['test_r2']:.3f}")
        st.metric("Test RMSE", f"{metadata['test_rmse']:,.0f} hg/ha")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">🧠 What Drove This Prediction</div>'
        '<div class="section-sub">Each bar shows how much that factor pushed this specific '
        'prediction up or down from the baseline, in hg/ha.</div>',
        unsafe_allow_html=True,
    )
    contrib = explain_prediction(model, metadata, input_fe, feature_cols)

    if contrib is not None:
        top = contrib.head(8).sort_values()
        colors = [PRIMARY_LIGHT if v >= 0 else "#EF5350" for v in top.values]
        fig_c = go.Figure(go.Bar(
            x=top.values, y=top.index, orientation="h",
            marker_color=colors,
            text=[f"{v:+,.0f}" for v in top.values], textposition="outside",
        ))
        fig_c.update_layout(
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=40, t=10, b=10), height=340,
            xaxis_title="Contribution to prediction (hg/ha)", yaxis_title="",
        )
        st.plotly_chart(fig_c, width='stretch', config={"displayModeBar": False})
        st.caption(
            "Positive bars pushed the prediction above the model's baseline average; "
            "negative bars pulled it below."
        )
    else:
        st.info(
            "Per-prediction breakdown is only available for the XGBoost model. "
            "Showing this model's overall (global) top feature importances instead.",
            icon="ℹ️",
        )
        importances = model.named_steps["model"].feature_importances_
        pre = model.named_steps["preprocess"]
        cat_names = pre.named_transformers_["cat"].get_feature_names_out(metadata["categorical_features"])
        all_names = metadata["numeric_features"] + list(cat_names)
        fi = pd.Series(importances, index=all_names).sort_values(ascending=False).head(8).sort_values()
        fig_fi = go.Figure(go.Bar(x=fi.values, y=fi.index, orientation="h", marker_color=PRIMARY_LIGHT))
        fig_fi.update_layout(
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10), height=340,
            xaxis_title="Importance", yaxis_title="",
        )
        st.plotly_chart(fig_fi, width='stretch', config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Fill in the form above and click **Predict Yield** to generate an estimate.", icon="🌱")

# ---------------------------------------------------------------------------
# Dataset explorer
# ---------------------------------------------------------------------------
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
    st.plotly_chart(fig1, width='stretch', config={"displayModeBar": False})

with tab2:
    top = ref.groupby("country")["yield_tonnes_per_ha"].median().sort_values(ascending=False).head(15)
    fig2 = go.Figure(go.Bar(
        x=top.values, y=top.index, orientation="h",
        marker_color=PRIMARY_LIGHT,
    ))
    fig2.update_layout(**plotly_layout, yaxis=dict(autorange="reversed"),
                        xaxis_title="Median Yield (tonnes/ha)", yaxis_title="")
    st.plotly_chart(fig2, width='stretch', config={"displayModeBar": False})

with tab3:
    yearly = ref.groupby("year")["yield_tonnes_per_ha"].mean().reset_index()
    fig3 = px.line(yearly, x="year", y="yield_tonnes_per_ha", markers=True)
    fig3.update_traces(line_color=ACCENT, marker=dict(size=6, color=PRIMARY_LIGHT))
    fig3.update_layout(**plotly_layout, xaxis_title="Year", yaxis_title="Avg Yield (tonnes/ha)")
    st.plotly_chart(fig3, width='stretch', config={"displayModeBar": False})

st.markdown('</div>', unsafe_allow_html=True)

with st.expander("🔎  Preview raw dataset"):
    st.dataframe(ref.head(50), width='stretch')

st.markdown(
    '<p class="footer-note">Model trained in notebooks/Crop_Yield_Prediction.ipynb · '
    'Served via Streamlit + FastAPI</p>',
    unsafe_allow_html=True,
)