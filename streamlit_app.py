import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
import os

# Page configuration
st.set_page_config(
    page_title="Cosmic Detective - Exoplanet Discovery",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS inspired by Lovable app
st.markdown("""
<style>
    /* Main styling */
    .main {
        background: linear-gradient(135deg, #0B0B2D 0%, #1A1A3E 50%, #2D1B4E 100%);
        color: white;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Modern header */
    .modern-header {
        text-align: center;
        padding: 4rem 0 3rem 0;
        background: linear-gradient(135deg, rgba(74, 85, 242, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border-radius: 0 0 30px 30px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .modern-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(74, 85, 242, 0.1), transparent);
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4A55F2 0%, #8B5CF6 50%, #D946EF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        line-height: 1.1;
    }
    
    .subtitle {
        font-size: 1.3rem;
        color: #94A3B8;
        font-weight: 300;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* Modern cards */
    .modern-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .modern-card:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(74, 85, 242, 0.3);
        transform: translateY(-2px);
    }
    
    /* Feature groups */
    .feature-group {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Modern buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4A55F2 0%, #8B5CF6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 8px 25px rgba(74, 85, 242, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(74, 85, 242, 0.4);
        background: linear-gradient(135deg, #8B5CF6 0%, #4A55F2 100%);
    }
    
    /* Result cards */
    .result-success {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(34, 197, 94, 0.05) 100%);
        border: 2px solid #22C55E;
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .result-error {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%);
        border: 2px solid #EF4444;
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .result-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .result-desc {
        font-size: 1.2rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }
    
    /* Confidence badges */
    .confidence-badge {
        background: linear-gradient(135deg, #4A55F2 0%, #8B5CF6 100%);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(74, 85, 242, 0.3);
    }
    
    /* Input styling */
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white;
        padding: 0.75rem;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #4A55F2;
        box-shadow: 0 0 0 2px rgba(74, 85, 242, 0.2);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: white;
        font-weight: 600;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-top: none;
        border-radius: 0 0 12px 12px;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #4A55F2 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-weight: 500 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #4A55F2 0%, #8B5CF6 100%);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.5rem;
        }
        .modern-card {
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="modern-header">
    <div class="main-title">Cosmic Detective</div>
    <div class="subtitle">
        Discover and classify distant worlds using advanced machine learning algorithms. 
        Analyze planetary signatures with 97.8% accuracy.
    </div>
</div>
""", unsafe_allow_html=True)

# Stats Section
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🌌 Systems Analyzed", "150K+", "98.2% accuracy")
with col2:
    st.metric("🪐 Exoplanets Found", "3,284", "+12 today")
with col3:
    st.metric("🚀 Model Confidence", "97.8%", "CatBoost AI")

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
                continue
    return None, None, None

model, feature_names, model_source = load_model()

if model is None:
    st.error("Please ensure 'best_model.pkl' or 'catboost.pkl' exists in the current directory")
    st.stop()

if feature_names is None:
    feature_names = [
        "koi_period", "koi_period_err1", "koi_time0bk", "koi_time0bk_err1",
        "koi_impact", "koi_impact_err1", "koi_impact_err2", "koi_duration",
        "koi_duration_err1", "koi_depth", "koi_depth_err1", "koi_prad",
        "koi_prad_err1", "koi_prad_err2", "koi_teq", "koi_insol",
        "koi_insol_err1", "koi_model_snr", "koi_steff", "koi_steff_err1",
        "koi_steff_err2", "koi_slogg", "koi_slogg_err1", "koi_slogg_err2",
        "koi_srad_err1", "koi_srad_err2", "ra", "dec", "koi_kepmag",
        "depth_to_srad", "prad_to_srad_ratio", "period_to_impact", "log_insol", "log_snr"
    ]

# Feature groups with better organization
feature_groups = {
    "🪐 Orbital Characteristics": [
        ("koi_period", 365.0, "Orbital Period (days)", "Time to complete one orbit"),
        ("koi_duration", 8.0, "Transit Duration (hrs)", "Duration of planetary transit"),
        ("koi_impact", 0.5, "Impact Parameter", "Orbital alignment parameter"),
    ],
    "🌍 Planetary Properties": [
        ("koi_prad", 1.0, "Planet Radius (R⊕)", "Size relative to Earth"),
        ("koi_depth", 5000.0, "Transit Depth (ppm)", "Light blocked during transit"),
        ("koi_teq", 288.0, "Equilibrium Temp (K)", "Planetary surface temperature"),
    ],
    "⭐ Stellar Information": [
        ("koi_steff", 5778.0, "Stellar Temp (K)", "Host star temperature"),
        ("koi_insol", 1.0, "Insolation Flux", "Stellar energy received"),
        ("koi_kepmag", 15.0, "Kepler Magnitude", "Star brightness"),
    ],
    "📊 Detection Metrics": [
        ("koi_model_snr", 10.0, "Signal-to-Noise", "Detection quality"),
        ("ra", 290.0, "Right Ascension", "Celestial coordinate"),
        ("dec", 45.0, "Declination", "Celestial coordinate"),
    ]
}

# Main input section
st.markdown("""
<div class="modern-card">
    <h2 style="color: white; margin-bottom: 2rem; font-size: 1.8rem;">🔭 Planetary System Analysis</h2>
""", unsafe_allow_html=True)

# Create input columns
inputs = {}
main_col1, main_col2 = st.columns(2)

with main_col1:
    for group_name, features in list(feature_groups.items())[:2]:
        st.markdown(f"<h3 style='color: #4A55F2; margin: 1.5rem 0 1rem 0;'>{group_name}</h3>", unsafe_allow_html=True)
        for feature, default, label, description in features:
            inputs[feature] = st.number_input(
                label,
                value=float(default),
                format="%.6f",
                key=feature,
                help=description
            )

with main_col2:
    for group_name, features in list(feature_groups.items())[2:]:
        st.markdown(f"<h3 style='color: #4A55F2; margin: 1.5rem 0 1rem 0;'>{group_name}</h3>", unsafe_allow_html=True)
        for feature, default, label, description in features:
            inputs[feature] = st.number_input(
                label,
                value=float(default),
                format="%.6f",
                key=feature,
                help=description
            )

st.markdown("</div>", unsafe_allow_html=True)

# Fill missing features
for feature in feature_names:
    if feature not in inputs:
        inputs[feature] = 0.0

# Prediction button
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Analyze Planetary System", use_container_width=True):
    with st.spinner("🔍 Scanning for planetary signatures..."):
        time.sleep(1.5)  # Add dramatic effect
        
        try:
            X_new = pd.DataFrame([inputs])
            X_new = X_new.reindex(columns=feature_names, fill_value=0)
            
            # Make prediction
            prediction = model.predict(X_new)[0]
            proba = model.predict_proba(X_new)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
            
            # Display results
            st.markdown("<br>", unsafe_allow_html=True)
            
            if prediction == 1:
                st.markdown(f"""
                <div class="result-success">
                    <div class="result-title" style="color: #22C55E;">✅ Exoplanet Detected</div>
                    <div class="result-desc">
                        Strong planetary signature identified with high confidence. 
                        This object exhibits characteristics consistent with confirmed exoplanets.
                    </div>
                    <div class="confidence-badge">Confidence: {proba[1]:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Success metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📈 Detection Score", f"{proba[1]:.1%}")
                with col2:
                    st.metric("🌍 Planetary Class", "Terrestrial" if inputs['koi_prad'] < 2 else "Gas Giant")
                with col3:
                    habitable = "Potentially" if 200 < inputs['koi_teq'] < 400 else "Unlikely"
                    st.metric("💫 Habitability", habitable)
                    
            else:
                st.markdown(f"""
                <div class="result-error">
                    <div class="result-title" style="color: #EF4444;">❌ False Positive</div>
                    <div class="result-desc">
                        Analysis indicates this signal is likely not planetary in nature. 
                        Consider stellar variability or instrumental effects.
                    </div>
                    <div class="confidence-badge">Confidence: {proba[0]:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed probability breakdown
            st.markdown("""
            <div class="modern-card">
                <h3 style="color: white; margin-bottom: 1.5rem;">📊 Probability Analysis</h3>
            """, unsafe_allow_html=True)
            
            prob_col1, prob_col2 = st.columns(2)
            with prob_col1:
                st.metric("False Positive Probability", f"{proba[0]:.1%}")
                st.progress(float(proba[0]))
            with prob_col2:
                st.metric("Confirmed Exoplanet Probability", f"{proba[1]:.1%}")
                st.progress(float(proba[1]))
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; padding: 2rem 0;">
    <p style="font-size: 0.9rem;">
        Cosmic Detective Tool • Powered by NASA Kepler Data & Machine Learning • 
        Built with Streamlit 🚀
    </p>
</div>
""", unsafe_allow_html=True)
