# streamlit_app.py
import os
import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(
    page_title="🚀 Exoplanet Classifier",
    page_icon="🪐",
    layout="wide"
)

# Custom CSS for better design
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    .stButton>button {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
        color: white;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 8px 16px rgba(139, 92, 246, 0.3);
        width: 100%;
    }
    .stButton>button:hover {
        box-shadow: 0 12px 24px rgba(139, 92, 246, 0.4);
        transform: translateY(-2px);
    }
    .success-box {
        background: rgba(34, 197, 94, 0.1);
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
    }
    .error-box {
        background: rgba(239, 68, 68, 0.1);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
    }
    h1 {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("# 🚀 Exoplanet Classifier")
st.markdown('<p class="subtitle">Discover and classify distant worlds using machine learning</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("⭐ Stars Analyzed", "150,000+")
with col2:
    st.metric("🪐 Exoplanets Found", "3,000+")
with col3:
    st.metric("🎯 Model Accuracy", "97.8%")

st.markdown("---")

# Load Model
@st.cache_resource
def load_model():
    model_files = ["best_model.pkl", "catboost.pkl", "model.pkl", "final_model.pkl"]
    for fname in model_files:
        if os.path.exists(fname):
            try:
                loaded = joblib.load(fname)
                if isinstance(loaded, tuple) and len(loaded) == 2:
                    return loaded[0], loaded[1], fname
                return loaded, None, fname
            except Exception as e:
                st.error(f"Error loading {fname}: {e}")
    return None, None, None

model, feature_names, model_source = load_model()

if model is None:
    st.error("⚠️ No model file found. Please upload 'best_model.pkl' or 'catboost.pkl'")
    st.stop()

if feature_names is None:
    feature_names = [
        "koi_period","koi_period_err1","koi_time0bk","koi_time0bk_err1",
        "koi_impact","koi_impact_err1","koi_impact_err2","koi_duration",
        "koi_duration_err1","koi_depth","koi_depth_err1","koi_prad",
        "koi_prad_err1","koi_prad_err2","koi_teq","koi_insol",
        "koi_insol_err1","koi_model_snr","koi_steff","koi_steff_err1",
        "koi_steff_err2","koi_slogg","koi_slogg_err1","koi_slogg_err2",
        "koi_srad_err1","koi_srad_err2","ra","dec","koi_kepmag",
        "depth_to_srad","prad_to_srad_ratio","period_to_impact","log_insol","log_snr"
    ]

st.success(f"✅ Model loaded: {model_source}")

# Feature Groups
feature_groups = {
    "🌍 Orbital Parameters": [
        ("koi_period", 1.0, "Orbital period (days)"),
        ("koi_period_err1", 0.0, "Period error"),
        ("koi_time0bk", 0.0, "Transit epoch"),
        ("koi_time0bk_err1", 0.0, "Transit epoch error"),
        ("koi_impact", 0.0, "Impact parameter"),
        ("koi_impact_err1", 0.0, "Impact error 1"),
        ("koi_impact_err2", 0.0, "Impact error 2"),
        ("koi_duration", 1.0, "Transit duration"),
        ("koi_duration_err1", 0.0, "Duration error"),
    ],
    "🪐 Planetary Properties": [
        ("koi_depth", 1.0, "Transit depth"),
        ("koi_depth_err1", 0.0, "Depth error"),
        ("koi_prad", 1.0, "Planetary radius"),
        ("koi_prad_err1", 0.0, "Radius error 1"),
        ("koi_prad_err2", 0.0, "Radius error 2"),
        ("koi_teq", 500.0, "Equilibrium temperature"),
        ("koi_insol", 1.0, "Insolation flux"),
        ("koi_insol_err1", 0.0, "Insolation error"),
        ("koi_model_snr", 0.0, "Signal-to-noise ratio"),
    ],
    "⭐ Stellar Properties": [
        ("koi_steff", 0.0, "Stellar effective temperature"),
        ("koi_steff_err1", 0.0, "Stellar temp error 1"),
        ("koi_steff_err2", 0.0, "Stellar temp error 2"),
        ("koi_slogg", 0.0, "Stellar surface gravity"),
        ("koi_slogg_err1", 0.0, "Surface gravity error 1"),
        ("koi_slogg_err2", 0.0, "Surface gravity error 2"),
        ("koi_srad_err1", 0.0, "Stellar radius error 1"),
        ("koi_srad_err2", 0.0, "Stellar radius error 2"),
        ("ra", 0.0, "Right ascension"),
        ("dec", 0.0, "Declination"),
        ("koi_kepmag", 0.0, "Kepler magnitude"),
    ],
    "📊 Derived Features": [
        ("depth_to_srad", 0.0, "Depth to stellar radius ratio"),
        ("prad_to_srad_ratio", 0.0, "Planet to star radius ratio"),
        ("period_to_impact", 0.0, "Period to impact ratio"),
        ("log_insol", 0.0, "Log insolation"),
        ("log_snr", 0.0, "Log signal-to-noise"),
    ]
}

# Input Form
inputs = {}
for group_name, features in feature_groups.items():
    with st.expander(f"{group_name} ({len(features)} features)", expanded=True):
        cols = st.columns(3)
        for idx, (feature, default, label) in enumerate(features):
            with cols[idx % 3]:
                inputs[feature] = st.number_input(
                    label,
                    value=default,
                    format="%.6f",
                    key=feature,
                    help=f"Feature: {feature}"
                )

st.markdown("---")

# Predict Button
if st.button("🔮 Classify Exoplanet"):
    with st.spinner("🌌 Analyzing planetary data..."):
        try:
            X_new = pd.DataFrame([inputs])
            X_new = X_new.reindex(columns=feature_names, fill_value=0)
            
            pred = model.predict(X_new)
            proba = model.predict_proba(X_new) if hasattr(model, "predict_proba") else None
            
            prediction = pred[0] if hasattr(pred, "__len__") else pred
            
            if prediction == 1 or str(prediction).upper() == "CONFIRMED":
                st.markdown("""
                <div class="success-box">
                    <h2>✅ CONFIRMED EXOPLANET</h2>
                    <p style="font-size: 1.2rem; color: #22c55e;">
                        This object shows strong evidence of being an exoplanet
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="error-box">
                    <h2>❌ FALSE POSITIVE</h2>
                    <p style="font-size: 1.2rem; color: #ef4444;">
                        This object is likely not an exoplanet
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            if proba is not None:
                st.markdown("### 📊 Prediction Confidence")
                proba_values = proba[0].tolist() if hasattr(proba[0], "tolist") else proba[0]
                conf_col1, conf_col2 = st.columns(2)
                with conf_col1:
                    st.metric("False Positive Probability", f"{proba_values[0]:.2%}")
                with conf_col2:
                    st.metric("Confirmed Probability", f"{proba_values[1]:.2%}")
                    
        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")
            st.write("Debug Info:")
            st.write("Input shape:", X_new.shape)
            st.write("Expected features:", len(feature_names))
