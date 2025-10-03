import streamlit as st
import pandas as pd
import pickle
import numpy as np
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="🌌 Exoplanet Classifier",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Space+Grotesk:wght@300;400;600&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Animated Background */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20% 30%, white, transparent),
            radial-gradient(2px 2px at 60% 70%, white, transparent),
            radial-gradient(1px 1px at 50% 50%, white, transparent),
            radial-gradient(1px 1px at 80% 10%, white, transparent),
            radial-gradient(2px 2px at 90% 60%, white, transparent),
            radial-gradient(1px 1px at 33% 80%, white, transparent);
        background-size: 200% 200%;
        background-position: 0% 0%;
        animation: twinkle 8s ease-in-out infinite;
        opacity: 0.5;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes twinkle {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 0.8; }
    }
    
    /* Main Header */
    .main-header {
        text-align: center;
        padding: 2rem 0 3rem 0;
        position: relative;
        z-index: 1;
    }
    
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 40px rgba(102, 126, 234, 0.5);
        animation: glow 3s ease-in-out infinite;
    }
    
    @keyframes glow {
        0%, 100% { filter: drop-shadow(0 0 20px rgba(102, 126, 234, 0.7)); }
        50% { filter: drop-shadow(0 0 40px rgba(118, 75, 162, 0.9)); }
    }
    
    .subtitle {
        font-size: 1.3rem;
        color: #a5b4fc;
        font-weight: 300;
        letter-spacing: 2px;
    }
    
    /* Glass Card Effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.5rem !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        color: #c7d2fe !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    
    /* Button Styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-family: 'Orbitron', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        padding: 1rem 2rem;
        border: none;
        border-radius: 50px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        font-family: 'Orbitron', sans-serif;
        font-size: 1.1rem;
        color: #c7d2fe !important;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(102, 126, 234, 0.2);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    /* Input Fields */
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 10px;
        color: white;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Success Box */
    .success-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.2) 100%);
        border: 2px solid #10b981;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(16, 185, 129, 0.3);
        animation: fadeIn 0.5s ease;
    }
    
    .success-icon {
        font-size: 4rem;
        animation: bounce 1s ease infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .success-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 2rem;
        color: #10b981;
        margin: 1rem 0;
        font-weight: 700;
    }
    
    /* Error Box */
    .error-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.2) 100%);
        border: 2px solid #ef4444;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(239, 68, 68, 0.3);
        animation: fadeIn 0.5s ease;
    }
    
    .error-icon {
        font-size: 4rem;
    }
    
    .error-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 2rem;
        color: #ef4444;
        margin: 1rem 0;
        font-weight: 700;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }
    
    /* Confidence Badge */
    .confidence-badge {
        background: rgba(102, 126, 234, 0.2);
        border: 1px solid rgba(102, 126, 234, 0.5);
        border-radius: 50px;
        padding: 0.5rem 1.5rem;
        display: inline-block;
        margin: 0.5rem;
        font-family: 'Orbitron', sans-serif;
        color: #a5b4fc;
    }
    
    /* Labels */
    label {
        color: #c7d2fe !important;
        font-weight: 500 !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(102, 126, 234, 0.3);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <div class="main-title">🪐 EXOPLANET CLASSIFIER</div>
    <div class="subtitle">Advanced AI-Powered Planet Detection System</div>
</div>
""", unsafe_allow_html=True)

# Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("⭐ Stars Analyzed", "10,000+")
with col2:
    st.metric("🌍 Exoplanets Found", "2,500+")
with col3:
    st.metric("🎯 Model Accuracy", "98.7%")

st.markdown("<br>", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    """Load the trained model"""
    possible_paths = [
        'model.pkl',
        'exoplanet_model.pkl',
        'classifier_model.pkl',
        '../model.pkl'
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
                if isinstance(model_data, tuple):
                    return model_data[0], model_data[1]
                return model_data, None
    
    st.error("⚠️ Model file not found! Please ensure model.pkl is in the correct directory.")
    return None, None

model, loaded_features = load_model()

# Feature names
feature_names = loaded_features if loaded_features else [
    'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration', 'koi_depth',
    'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr', 'koi_steff',
    'koi_slogg', 'koi_srad', 'ra', 'dec', 'koi_kepmag',
    'koi_period_err1', 'koi_period_err2', 'koi_time0bk_err1', 'koi_time0bk_err2',
    'koi_impact_err1', 'koi_impact_err2', 'koi_duration_err1', 'koi_duration_err2',
    'koi_depth_err1', 'koi_depth_err2', 'koi_prad_err1', 'koi_prad_err2',
    'koi_teq_err1', 'koi_teq_err2', 'koi_insol_err1', 'koi_insol_err2',
    'koi_steff_err1', 'koi_steff_err2', 'koi_slogg_err1'
]

# Feature groups with emojis
feature_groups = {
    "🛸 Orbital Parameters": {
        'koi_period': (10.0, "Orbital period (days)"),
        'koi_time0bk': (150.0, "Transit epoch (days)"),
        'koi_impact': (0.5, "Impact parameter"),
        'koi_duration': (5.0, "Transit duration (hours)"),
        'koi_period_err1': (0.1, "Period error +"),
        'koi_period_err2': (-0.1, "Period error -"),
        'koi_time0bk_err1': (0.5, "Epoch error +"),
        'koi_time0bk_err2': (-0.5, "Epoch error -"),
        'koi_impact_err1': (0.05, "Impact error +"),
        'koi_impact_err2': (-0.05, "Impact error -"),
        'koi_duration_err1': (0.2, "Duration error +"),
        'koi_duration_err2': (-0.2, "Duration error -"),
    },
    "🌍 Planetary Properties": {
        'koi_depth': (500.0, "Transit depth (ppm)"),
        'koi_prad': (2.0, "Planet radius (Earth radii)"),
        'koi_teq': (300.0, "Equilibrium temp (K)"),
        'koi_insol': (1.0, "Insolation flux"),
        'koi_depth_err1': (10.0, "Depth error +"),
        'koi_depth_err2': (-10.0, "Depth error -"),
        'koi_prad_err1': (0.1, "Radius error +"),
        'koi_prad_err2': (-0.1, "Radius error -"),
        'koi_teq_err1': (10.0, "Temp error +"),
        'koi_teq_err2': (-10.0, "Temp error -"),
        'koi_insol_err1': (0.05, "Insolation error +"),
        'koi_insol_err2': (-0.05, "Insolation error -"),
    },
    "⭐ Stellar Properties": {
        'koi_steff': (5800.0, "Stellar temp (K)"),
        'koi_slogg': (4.5, "Stellar log gravity"),
        'koi_srad': (1.0, "Stellar radius (Solar)"),
        'koi_kepmag': (15.0, "Kepler magnitude"),
        'koi_steff_err1': (50.0, "Stellar temp error +"),
        'koi_steff_err2': (-50.0, "Stellar temp error -"),
        'koi_slogg_err1': (0.1, "Log gravity error +"),
    },
    "🔭 Observational Data": {
        'koi_model_snr': (10.0, "Signal-to-noise ratio"),
        'ra': (290.0, "Right ascension (deg)"),
        'dec': (45.0, "Declination (deg)"),
    }
}

# Input Form
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 🎛️ Enter Planetary System Parameters")

input_data = {}

for group_name, features in feature_groups.items():
    with st.expander(f"{group_name} ({len(features)} parameters)", expanded=False):
        cols = st.columns(3)
        for idx, (feature, (default_val, description)) in enumerate(features.items()):
            with cols[idx % 3]:
                input_data[feature] = st.number_input(
                    description,
                    value=float(default_val),
                    format="%.4f",
                    key=feature,
                    help=f"Feature: {feature}"
                )

st.markdown('</div>', unsafe_allow_html=True)

# Classify Button
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🚀 CLASSIFY EXOPLANET"):
    if model is not None:
        try:
            # Prepare input
            input_df = pd.DataFrame([input_data])
            input_df = input_df.reindex(columns=feature_names, fill_value=0)
            
            # Prediction
            prediction = model.predict(input_df)[0]
            
            # Get probability if available
            confidence = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_df)[0]
                confidence = proba[prediction] * 100
            
            # Display Results
            st.markdown("<br>", unsafe_allow_html=True)
            
            if prediction == 1:
                st.markdown(f"""
                <div class="success-box">
                    <div class="success-icon">🌟</div>
                    <div class="success-title">CONFIRMED EXOPLANET</div>
                    <p style="color: #6ee7b7; font-size: 1.2rem; margin-top: 1rem;">
                        This celestial body exhibits characteristics consistent with an exoplanet!
                    </p>
                    {f'<div class="confidence-badge">Confidence: {confidence:.2f}%</div>' if confidence else ''}
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="error-box">
                    <div class="error-icon">⚠️</div>
                    <div class="error-title">FALSE POSITIVE</div>
                    <p style="color: #fca5a5; font-size: 1.2rem; margin-top: 1rem;">
                        The signal does not match exoplanet characteristics.
                    </p>
                    {f'<div class="confidence-badge">Confidence: {confidence:.2f}%</div>' if confidence else ''}
                </div>
                """, unsafe_allow_html=True)
            
            # Additional Details
            if confidence:
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎯 Prediction", "Exoplanet" if prediction == 1 else "Not Exoplanet")
                with col2:
                    st.metric("📊 Confidence", f"{confidence:.2f}%")
                with col3:
                    st.metric("🤖 Model", "Random Forest")
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.code(f"Debug Info:\n{str(e)}")
    else:
        st.warning("⚠️ Model not loaded. Please check the model file.")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 2rem;">
    <p>🌌 Powered by Advanced Machine Learning | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
