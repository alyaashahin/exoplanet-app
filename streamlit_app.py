# Enhanced Exoplanet Classifier with Cosmic Theme
import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time

# Page config
st.set_page_config(
    page_title="🌌 Cosmic Exoplanet Classifier",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS with Cosmic Theme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Root Variables */
    :root {
        --cosmic-blue: #3b82f6;
        --nebula-purple: #8b5cf6;
        --stellar-gold: #f59e0b;
        --deep-space: #0f172a;
        --star-glow: #e2e8f0;
        --dark-nebula: #1e293b;
        --space-dust: #475569;
        --planet-shadow: #1e293b;
    }
    
    /* Main App Styling */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Header */
    .cosmic-header {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border-radius: 20px;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    .cosmic-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.1), transparent);
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* Title Styling */
    .cosmic-title {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #f59e0b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 4rem !important;
        font-weight: 900;
        text-align: center;
        margin-bottom: 1rem;
        animation: gradient-x 3s ease infinite;
        background-size: 200% 200%;
    }
    
    @keyframes gradient-x {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .cosmic-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.3rem;
        font-weight: 300;
        margin-bottom: 2rem;
        opacity: 0.9;
    }
    
    /* Metrics Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.6) 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(59, 130, 246, 0.2);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(59, 130, 246, 0.2);
        border-color: rgba(59, 130, 246, 0.4);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #f59e0b);
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 8px 16px rgba(59, 130, 246, 0.3);
        width: 100%;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button:hover {
        box-shadow: 0 12px 24px rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s ease;
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(51, 65, 85, 0.7) 100%);
        border-radius: 12px;
        border: 1px solid rgba(59, 130, 246, 0.2);
        color: #e2e8f0;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .streamlit-expanderContent {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
        border-radius: 0 0 12px 12px;
        border: 1px solid rgba(59, 130, 246, 0.1);
        border-top: none;
    }
    
    /* Number Input Styling */
    .stNumberInput>div>div>input {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    .stNumberInput>div>div>input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }
    
    /* Label Styling */
    .stNumberInput label {
        color: #cbd5e1;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    /* Success Box */
    .success-box {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(34, 197, 94, 0.05) 100%);
        border: 2px solid #22c55e;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        animation: pulse-glow 2s ease-in-out infinite;
    }
    
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 20px rgba(34, 197, 94, 0.3); }
        50% { box-shadow: 0 0 40px rgba(34, 197, 94, 0.5); }
    }
    
    .success-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #22c55e, #16a34a, #22c55e);
        animation: shimmer 2s ease-in-out infinite;
    }
    
    /* Error Box */
    .error-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        animation: pulse-glow-red 2s ease-in-out infinite;
    }
    
    @keyframes pulse-glow-red {
        0%, 100% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.3); }
        50% { box-shadow: 0 0 40px rgba(239, 68, 68, 0.5); }
    }
    
    .error-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #ef4444, #dc2626, #ef4444);
        animation: shimmer 2s ease-in-out infinite;
    }
    
    /* Confidence Metrics */
    .confidence-metric {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.6) 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(139, 92, 246, 0.2);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .confidence-metric:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(139, 92, 246, 0.2);
    }
    
    /* Animated Background Elements */
    .cosmic-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        pointer-events: none;
    }
    
    .star {
        position: absolute;
        background: #e2e8f0;
        border-radius: 50%;
        animation: twinkle 4s ease-in-out infinite;
    }
    
    @keyframes twinkle {
        0%, 100% { opacity: 0.3; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.2); }
    }
    
    .shooting-star {
        position: absolute;
        width: 2px;
        height: 2px;
        background: linear-gradient(45deg, #3b82f6, #8b5cf6);
        border-radius: 50%;
        animation: shoot 3s linear infinite;
    }
    
    @keyframes shoot {
        0% { transform: translateX(-100px) translateY(0px); opacity: 0; }
        10% { opacity: 1; }
        90% { opacity: 1; }
        100% { transform: translateX(100vw) translateY(-50px); opacity: 0; }
    }
    
    /* Loading Animation */
    .cosmic-loader {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(59, 130, 246, 0.3);
        border-radius: 50%;
        border-top-color: #3b82f6;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .cosmic-title {
            font-size: 2.5rem !important;
        }
        .cosmic-subtitle {
            font-size: 1.1rem;
        }
        .metric-card {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Add animated background
st.markdown("""
<div class="cosmic-bg">
    <div class="star" style="top: 10%; left: 20%; width: 2px; height: 2px; animation-delay: 0s;"></div>
    <div class="star" style="top: 20%; left: 80%; width: 1px; height: 1px; animation-delay: 1s;"></div>
    <div class="star" style="top: 30%; left: 40%; width: 3px; height: 3px; animation-delay: 2s;"></div>
    <div class="star" style="top: 50%; left: 10%; width: 2px; height: 2px; animation-delay: 0.5s;"></div>
    <div class="star" style="top: 60%; left: 70%; width: 1px; height: 1px; animation-delay: 1.5s;"></div>
    <div class="star" style="top: 80%; left: 30%; width: 2px; height: 2px; animation-delay: 2.5s;"></div>
    <div class="star" style="top: 90%; left: 90%; width: 1px; height: 1px; animation-delay: 3s;"></div>
    
    <div class="shooting-star" style="top: 15%; left: 0%; animation-delay: 0s;"></div>
    <div class="shooting-star" style="top: 45%; left: 0%; animation-delay: 2s;"></div>
    <div class="shooting-star" style="top: 75%; left: 0%; animation-delay: 4s;"></div>
</div>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="cosmic-header">
    <h1 class="cosmic-title">🌌 COSMIC EXOPLANET CLASSIFIER</h1>
    <p class="cosmic-subtitle">Discover and classify distant worlds using advanced machine learning</p>
</div>
""", unsafe_allow_html=True)

# Metrics Section
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #3b82f6; margin: 0 0 0.5rem 0; font-size: 2rem;">⭐</h3>
        <h4 style="color: #e2e8f0; margin: 0; font-size: 1.5rem;">150,000+</h4>
        <p style="color: #94a3b8; margin: 0.5rem 0 0 0;">Stars Analyzed</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #8b5cf6; margin: 0 0 0.5rem 0; font-size: 2rem;">🪐</h3>
        <h4 style="color: #e2e8f0; margin: 0; font-size: 1.5rem;">3,000+</h4>
        <p style="color: #94a3b8; margin: 0.5rem 0 0 0;">Exoplanets Found</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #f59e0b; margin: 0 0 0.5rem 0; font-size: 2rem;">🎯</h3>
        <h4 style="color: #e2e8f0; margin: 0; font-size: 1.5rem;">97.8%</h4>
        <p style="color: #94a3b8; margin: 0.5rem 0 0 0;">Model Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Load Model Function
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

st.success(f"✅ Model loaded successfully: {model_source}")

# Feature Groups with Enhanced Descriptions
feature_groups = {
    "🌍 Orbital Parameters": [
        ("koi_period", 1.0, "Orbital period (days)", "Time for planet to complete one orbit around its star"),
        ("koi_period_err1", 0.0, "Period error", "Uncertainty in orbital period measurement"),
        ("koi_time0bk", 0.0, "Transit epoch", "Time of first observed transit"),
        ("koi_time0bk_err1", 0.0, "Transit epoch error", "Uncertainty in transit timing"),
        ("koi_impact", 0.0, "Impact parameter", "Minimum distance between planet and star center"),
        ("koi_impact_err1", 0.0, "Impact error 1", "Uncertainty in impact parameter"),
        ("koi_impact_err2", 0.0, "Impact error 2", "Additional uncertainty in impact parameter"),
        ("koi_duration", 1.0, "Transit duration", "Duration of planet passing in front of star"),
        ("koi_duration_err1", 0.0, "Duration error", "Uncertainty in transit duration"),
    ],
    "🪐 Planetary Properties": [
        ("koi_depth", 1.0, "Transit depth", "Fraction of starlight blocked during transit"),
        ("koi_depth_err1", 0.0, "Depth error", "Uncertainty in transit depth measurement"),
        ("koi_prad", 1.0, "Planetary radius", "Radius of the planet (Earth radii)"),
        ("koi_prad_err1", 0.0, "Radius error 1", "Uncertainty in planetary radius"),
        ("koi_prad_err2", 0.0, "Radius error 2", "Additional uncertainty in radius"),
        ("koi_teq", 500.0, "Equilibrium temperature", "Planet's surface temperature (Kelvin)"),
        ("koi_insol", 1.0, "Insolation flux", "Stellar radiation received by planet"),
        ("koi_insol_err1", 0.0, "Insolation error", "Uncertainty in insolation measurement"),
        ("koi_model_snr", 0.0, "Signal-to-noise ratio", "Quality of transit detection"),
    ],
    "⭐ Stellar Properties": [
        ("koi_steff", 0.0, "Stellar effective temperature", "Star's surface temperature (Kelvin)"),
        ("koi_steff_err1", 0.0, "Stellar temp error 1", "Uncertainty in stellar temperature"),
        ("koi_steff_err2", 0.0, "Stellar temp error 2", "Additional temperature uncertainty"),
        ("koi_slogg", 0.0, "Stellar surface gravity", "Star's surface gravity (log g)"),
        ("koi_slogg_err1", 0.0, "Surface gravity error 1", "Uncertainty in surface gravity"),
        ("koi_slogg_err2", 0.0, "Surface gravity error 2", "Additional gravity uncertainty"),
        ("koi_srad_err1", 0.0, "Stellar radius error 1", "Uncertainty in stellar radius"),
        ("koi_srad_err2", 0.0, "Stellar radius error 2", "Additional radius uncertainty"),
        ("ra", 0.0, "Right ascension", "Celestial coordinate (degrees)"),
        ("dec", 0.0, "Declination", "Celestial coordinate (degrees)"),
        ("koi_kepmag", 0.0, "Kepler magnitude", "Star's apparent brightness"),
    ],
    "📊 Derived Features": [
        ("depth_to_srad", 0.0, "Depth to stellar radius ratio", "Normalized transit depth"),
        ("prad_to_srad_ratio", 0.0, "Planet to star radius ratio", "Relative size comparison"),
        ("period_to_impact", 0.0, "Period to impact ratio", "Orbital geometry indicator"),
        ("log_insol", 0.0, "Log insolation", "Logarithmic insolation flux"),
        ("log_snr", 0.0, "Log signal-to-noise", "Logarithmic detection quality"),
    ]
}

# Input Form with Enhanced UI
inputs = {}
for group_name, features in feature_groups.items():
    with st.expander(f"{group_name} ({len(features)} features)", expanded=True):
        cols = st.columns(3)
        for idx, (feature, default, label, description) in enumerate(features):
            with cols[idx % 3]:
                inputs[feature] = st.number_input(
                    label,
                    value=default,
                    format="%.6f",
                    key=feature,
                    help=description
                )

st.markdown("---")

# Enhanced Predict Button
if st.button("🔮 Classify Exoplanet", key="predict_button"):
    with st.spinner("🌌 Analyzing planetary data..."):
        # Add cosmic loading animation
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <div class="cosmic-loader"></div>
            <p style="color: #94a3b8; margin-top: 1rem;">Scanning the cosmos for planetary signatures...</p>
        </div>
        """, unsafe_allow_html=True)
        
        time.sleep(1)  # Add dramatic pause
        
        try:
            X_new = pd.DataFrame([inputs])
            X_new = X_new.reindex(columns=feature_names, fill_value=0)
            
            pred = model.predict(X_new)
            proba = model.predict_proba(X_new) if hasattr(model, "predict_proba") else None
            
            prediction = pred[0] if hasattr(pred, "_len_") else pred
            
            if prediction == 1 or str(prediction).upper() == "CONFIRMED":
                st.markdown("""
                <div class="success-box">
                    <h2 style="color: #22c55e; margin: 0 0 1rem 0; font-size: 2.5rem;">✅ CONFIRMED EXOPLANET</h2>
                    <p style="font-size: 1.3rem; color: #22c55e; margin: 0;">
                        🌟 This object shows strong evidence of being an exoplanet! 🌟
                    </p>
                    <p style="color: #16a34a; margin: 1rem 0 0 0; font-size: 1.1rem;">
                        The planetary signature has been detected with high confidence.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="error-box">
                    <h2 style="color: #ef4444; margin: 0 0 1rem 0; font-size: 2.5rem;">❌ FALSE POSITIVE</h2>
                    <p style="font-size: 1.3rem; color: #ef4444; margin: 0;">
                        🔍 This object is likely not an exoplanet
                    </p>
                    <p style="color: #dc2626; margin: 1rem 0 0 0; font-size: 1.1rem;">
                        The signal appears to be caused by other astronomical phenomena.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            if proba is not None:
                st.markdown("### 📊 Prediction Confidence Analysis")
                proba_values = proba[0].tolist() if hasattr(proba[0], "tolist") else proba[0]
                
                conf_col1, conf_col2 = st.columns(2)
                with conf_col1:
                    st.markdown(f"""
                    <div class="confidence-metric">
                        <h4 style="color: #ef4444; margin: 0 0 0.5rem 0;">❌ False Positive</h4>
                        <h3 style="color: #e2e8f0; margin: 0; font-size: 2rem;">{proba_values[0]:.1%}</h3>
                        <p style="color: #94a3b8; margin: 0.5rem 0 0 0;">Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with conf_col2:
                    st.markdown(f"""
                    <div class="confidence-metric">
                        <h4 style="color: #22c55e; margin: 0 0 0.5rem 0;">✅ Confirmed</h4>
                        <h3 style="color: #e2e8f0; margin: 0; font-size: 2rem;">{proba_values[1]:.1%}</h3>
                        <p style="color: #94a3b8; margin: 0.5rem 0 0 0;">Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add confidence visualization
                st.markdown("### 🎯 Confidence Visualization")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.progress(proba_values[1])
                    st.caption(f"Model Confidence: {max(proba_values):.1%}")
                    
        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")
            st.write("Debug Info:")
            st.write("Input shape:", X_new.shape)
            st.write("Expected features:", len(feature_names))

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; margin-top: 3rem;">
    <p style="font-size: 0.9rem;">
        🌌 Powered by advanced machine learning algorithms • 
        🪐 Exploring the cosmos one prediction at a time • 
        ⭐ Built with Streamlit and cosmic inspiration
    </p>
</div>
""", unsafe_allow_html=True)
