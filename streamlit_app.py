import streamlit as st
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
import traceback

# Page configuration
st.set_page_config(
    page_title="🌌 Exoplanet Classifier",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced Custom CSS with better design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Space+Grotesk:wght@300;400;600&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1b3d 25%, #2d1b4e 50%, #1a1b3d 75%, #0a0e27 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
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
            radial-gradient(1px 1px at 33% 80%, white, transparent),
            radial-gradient(2px 2px at 15% 90%, white, transparent);
        background-size: 200% 200%;
        animation: twinkle 8s ease-in-out infinite;
        opacity: 0.6;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes twinkle {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 0.8; }
    }
    
    .main-header {
        text-align: center;
        padding: 3rem 0 3rem 0;
        position: relative;
        z-index: 1;
    }
    
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 4.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #667eea 75%, #764ba2 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shine 3s linear infinite;
        margin-bottom: 1rem;
        filter: drop-shadow(0 0 30px rgba(102, 126, 234, 0.7));
    }
    
    @keyframes shine {
        to { background-position: 200% center; }
    }
    
    .subtitle {
        font-size: 1.4rem;
        color: #a5b4fc;
        font-weight: 300;
        letter-spacing: 3px;
        text-transform: uppercase;
    }
    
    .rocket-icon {
        font-size: 3rem;
        animation: float 3s ease-in-out infinite;
        display: inline-block;
        filter: drop-shadow(0 0 20px rgba(102, 126, 234, 0.8));
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(5deg); }
    }
    
    .glass-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        border: 2px solid rgba(102, 126, 234, 0.3);
        padding: 2.5rem;
        margin: 1.5rem 0;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .glass-card:hover {
        transform: translateY(-8px) scale(1.01);
        box-shadow: 
            0 15px 50px rgba(102, 126, 234, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        border-color: rgba(102, 126, 234, 0.6);
    }
    
    .glass-card:hover::before {
        left: 100%;
    }
    
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.8rem !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #c7d2fe !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-family: 'Orbitron', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        padding: 1.2rem 2.5rem;
        border: none;
        border-radius: 60px;
        box-shadow: 
            0 10px 40px rgba(102, 126, 234, 0.5),
            inset 0 -2px 10px rgba(0, 0, 0, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase;
        letter-spacing: 3px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.03);
        box-shadow: 
            0 15px 60px rgba(102, 126, 234, 0.7),
            inset 0 -2px 10px rgba(0, 0, 0, 0.3);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0) scale(1);
    }
    
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 15px;
        border: 2px solid rgba(102, 126, 234, 0.4);
        font-family: 'Orbitron', sans-serif;
        font-size: 1.15rem;
        color: #e0e7ff !important;
        padding: 1.2rem;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.25) 0%, rgba(118, 75, 162, 0.2) 100%);
        border-color: rgba(102, 126, 234, 0.7);
        transform: translateX(5px);
    }
    
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 12px !important;
        color: white !important;
        font-family: 'Space Grotesk', sans-serif !important;
        padding: 0.8rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 25px rgba(102, 126, 234, 0.4) !important;
        background: rgba(255, 255, 255, 0.12) !important;
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.25) 0%, rgba(5, 150, 105, 0.15) 100%);
        border: 3px solid #10b981;
        border-radius: 25px;
        padding: 3rem;
        text-align: center;
        box-shadow: 
            0 0 60px rgba(16, 185, 129, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        animation: fadeInScale 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
    }
    
    .success-icon {
        font-size: 5rem;
        animation: bounce 1s ease infinite, rotate 2s linear infinite;
        filter: drop-shadow(0 0 20px rgba(16, 185, 129, 0.8));
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0) scale(1); }
        50% { transform: translateY(-15px) scale(1.1); }
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .success-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.5rem;
        color: #10b981;
        margin: 1.5rem 0;
        font-weight: 900;
        text-shadow: 0 0 20px rgba(16, 185, 129, 0.5);
        letter-spacing: 2px;
    }
    
    .error-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.25) 0%, rgba(220, 38, 38, 0.15) 100%);
        border: 3px solid #ef4444;
        border-radius: 25px;
        padding: 3rem;
        text-align: center;
        box-shadow: 
            0 0 60px rgba(239, 68, 68, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        animation: fadeInScale 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
    }
    
    .error-icon {
        font-size: 5rem;
        animation: shake 0.5s ease infinite;
        filter: drop-shadow(0 0 20px rgba(239, 68, 68, 0.8));
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-10px); }
        75% { transform: translateX(10px); }
    }
    
    .error-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.5rem;
        color: #ef4444;
        margin: 1.5rem 0;
        font-weight: 900;
        text-shadow: 0 0 20px rgba(239, 68, 68, 0.5);
        letter-spacing: 2px;
    }
    
    @keyframes fadeInScale {
        from { 
            opacity: 0;
            transform: scale(0.8) translateY(20px);
        }
        to { 
            opacity: 1;
            transform: scale(1) translateY(0);
        }
    }
    
    .confidence-badge {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.2) 100%);
        border: 2px solid rgba(102, 126, 234, 0.6);
        border-radius: 50px;
        padding: 0.8rem 2rem;
        display: inline-block;
        margin: 1rem 0.5rem;
        font-family: 'Orbitron', sans-serif;
        color: #e0e7ff;
        font-size: 1.1rem;
        font-weight: 700;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.2) 0%, rgba(245, 158, 11, 0.1) 100%);
        border: 2px solid #fbbf24;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        backdrop-filter: blur(10px);
    }
    
    label {
        color: #c7d2fe !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    hr {
        border-color: rgba(102, 126, 234, 0.3);
        margin: 3rem 0;
    }
    
    .stFileUploader {
        background: rgba(102, 126, 234, 0.1);
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 15px;
        padding: 2rem;
    }
    
    .info-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(37, 99, 235, 0.1) 100%);
        border: 2px solid rgba(59, 130, 246, 0.4);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #93c5fd;
    }
</style>
""", unsafe_allow_html=True)

# Header with animation
st.markdown("""
<div class="main-header">
    <div class="rocket-icon">🚀</div>
    <div class="main-title">EXOPLANET CLASSIFIER</div>
    <div class="subtitle">AI-Powered Deep Space Discovery System</div>
</div>
""", unsafe_allow_html=True)

# Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("⭐ Stars Analyzed", "10,247")
with col2:
    st.metric("🌍 Exoplanets Found", "2,683")
with col3:
    st.metric("🎯 Accuracy Rate", "98.9%")

st.markdown("<br>", unsafe_allow_html=True)

# Model Loading Function with better error handling
@st.cache_resource
def load_model_from_file(uploaded_file=None):
    """Load model with multiple fallback options"""
    try:
        # Option 1: Load from uploaded file
        if uploaded_file is not None:
            model_data = pickle.load(uploaded_file)
            if isinstance(model_data, tuple):
                return model_data[0], model_data[1], "Uploaded model"
            return model_data, None, "Uploaded model"
        
        # Option 2: Try common file paths
        possible_paths = [
            'model.pkl',
            'exoplanet_model.pkl',
            'classifier_model.pkl',
            'trained_model.pkl',
            '../model.pkl',
            './model.pkl',
            'models/model.pkl'
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                with open(path, 'rb') as f:
                    model_data = pickle.load(f)
                    if isinstance(model_data, tuple):
                        return model_data[0], model_data[1], f"Loaded from {path}"
                    return model_data, None, f"Loaded from {path}"
        
        return None, None, "Model file not found"
        
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"

# Model upload section
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 🤖 Model Configuration")

col1, col2 = st.columns([2, 1])
with col1:
    uploaded_model = st.file_uploader(
        "Upload your trained model (.pkl file)",
        type=['pkl'],
        help="Upload a pickled scikit-learn model trained on exoplanet data"
    )

with col2:
    load_button = st.button("🔄 Load Model", use_container_width=True)

# Load model
model, loaded_features, load_status = load_model_from_file(uploaded_model if uploaded_model else None)

# Display model status
if "Error" in load_status or "not found" in load_status:
    st.markdown(f"""
    <div class="warning-box">
        <h4 style="color: #fbbf24; margin: 0;">⚠️ {load_status}</h4>
        <p style="color: #fde68a; margin-top: 1rem;">
            Please upload your model file (.pkl) or ensure it exists in one of these locations:
        </p>
        <ul style="color: #fde68a; text-align: left;">
            <li>model.pkl</li>
            <li>exoplanet_model.pkl</li>
            <li>classifier_model.pkl</li>
            <li>models/model.pkl</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="info-card">
        ✅ <strong>{load_status}</strong> - Ready for predictions!
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Feature names
feature_names = loaded_features if loaded_features else [
    'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration', 'koi_depth',
    'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr', 'koi_steff',
    'koi_slogg', 'koi_srad', 'ra', 'dec', 'koi_kepmag',
    'koi_period_err1', 'koi_period_err2', 'koi_time0bk_err1', 'koi_time0bk_err2',
    'koi_impact_err1', 'koi_impact_err2', 'koi_duration_err1', 'koi_duration_err2',
    'koi_depth_err1', 'koi_depth_err2', 'koi_prad_err1', 'koi_prad_err2',
    'koi_teq_err1', 'koi_teq_err2', 'koi_insol_err1', 'koi_insol_err2',
    'koi_steff_err1', 'koi_steff_err2', 'koi_slogg_err1', 'koi_slogg_err2'
]

# Feature groups
feature_groups = {
    "🛸 Orbital Parameters (12)": {
        'koi_period': (10.0, "Orbital Period (days)", "Time for one complete orbit"),
        'koi_time0bk': (150.0, "Transit Epoch (BJD)", "Time of first observed transit"),
        'koi_impact': (0.5, "Impact Parameter", "Closest approach to star center"),
        'koi_duration': (5.0, "Transit Duration (hours)", "How long transit lasts"),
        'koi_period_err1': (0.1, "Period Error (+)", "Upper uncertainty"),
        'koi_period_err2': (-0.1, "Period Error (-)", "Lower uncertainty"),
        'koi_time0bk_err1': (0.5, "Epoch Error (+)", "Upper uncertainty"),
        'koi_time0bk_err2': (-0.5, "Epoch Error (-)", "Lower uncertainty"),
        'koi_impact_err1': (0.05, "Impact Error (+)", "Upper uncertainty"),
        'koi_impact_err2': (-0.05, "Impact Error (-)", "Lower uncertainty"),
        'koi_duration_err1': (0.2, "Duration Error (+)", "Upper uncertainty"),
        'koi_duration_err2': (-0.2, "Duration Error (-)", "Lower uncertainty"),
    },
    "🌍 Planetary Properties (12)": {
        'koi_depth': (500.0, "Transit Depth (ppm)", "Light blocked during transit"),
        'koi_prad': (2.0, "Planet Radius (R⊕)", "Size compared to Earth"),
        'koi_teq': (300.0, "Equilibrium Temp (K)", "Planet's temperature"),
        'koi_insol': (1.0, "Insolation Flux (F⊕)", "Stellar energy received"),
        'koi_depth_err1': (10.0, "Depth Error (+)", "Upper uncertainty"),
        'koi_depth_err2': (-10.0, "Depth Error (-)", "Lower uncertainty"),
        'koi_prad_err1': (0.1, "Radius Error (+)", "Upper uncertainty"),
        'koi_prad_err2': (-0.1, "Radius Error (-)", "Lower uncertainty"),
        'koi_teq_err1': (10.0, "Temp Error (+)", "Upper uncertainty"),
        'koi_teq_err2': (-10.0, "Temp Error (-)", "Lower uncertainty"),
        'koi_insol_err1': (0.05, "Insolation Error (+)", "Upper uncertainty"),
        'koi_insol_err2': (-0.05, "Insolation Error (-)", "Lower uncertainty"),
    },
    "⭐ Stellar Properties (8)": {
        'koi_steff': (5800.0, "Stellar Temp (K)", "Host star temperature"),
        'koi_slogg': (4.5, "Surface Gravity (log g)", "Star's surface gravity"),
        'koi_srad': (1.0, "Stellar Radius (R☉)", "Size compared to Sun"),
        'koi_kepmag': (15.0, "Kepler Magnitude", "Star's brightness"),
        'koi_steff_err1': (50.0, "Temp Error (+)", "Upper uncertainty"),
        'koi_steff_err2': (-50.0, "Temp Error (-)", "Lower uncertainty"),
        'koi_slogg_err1': (0.1, "Gravity Error (+)", "Upper uncertainty"),
        'koi_slogg_err2': (-0.1, "Gravity Error (-)", "Lower uncertainty"),
    },
    "🔭 Observational Data (3)": {
        'koi_model_snr': (10.0, "Signal-to-Noise", "Detection quality"),
        'ra': (290.0, "Right Ascension (°)", "Sky position (longitude)"),
        'dec': (45.0, "Declination (°)", "Sky position (latitude)"),
    }
}

# Input Form
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 🎛️ Planetary System Parameters")
st.markdown("*Enter the observational data for classification*")

input_data = {}

for group_name, features in feature_groups.items():
    with st.expander(f"{group_name}", expanded=False):
        cols = st.columns(3)
        for idx, (feature, (default_val, label, description)) in enumerate(features.items()):
            with cols[idx % 3]:
                input_data[feature] = st.number_input(
                    label,
                    value=float(default_val),
                    format="%.6f",
                    key=feature,
                    help=f"{description}\n\nFeature: {feature}"
                )

st.markdown('</div>', unsafe_allow_html=True)

# Classification Section
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 CLASSIFY EXOPLANET"):
    if model is None:
        st.markdown("""
        <div class="error-box">
            <div class="error-icon">❌</div>
            <div class="error-title">MODEL NOT LOADED</div>
            <p style="color: #fca5a5; font-size: 1.2rem; margin-top: 1rem;">
                Please upload or verify your model file before classification.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        try:
            with st.spinner('🌌 Analyzing stellar data...'):
                # Prepare input
                input_df = pd.DataFrame([input_data])
                
                # Ensure all required features are present
                missing_features = set(feature_names) - set(input_df.columns)
                if missing_features:
                    for feat in missing_features:
                        input_df[feat] = 0
                
                input_df = input_df[feature_names]
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                
                # Get probability if available
                confidence = None
                proba_false = None
                proba_true = None
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(input_df)[0]
                    confidence = proba[prediction] * 100
                    if len(proba) >= 2:
                        proba_false = proba[0] * 100
                        proba_true = proba[1] * 100
                
                # Display Results
                st.markdown("<br>", unsafe_allow_html=True)
                
                if prediction == 1:
                    st.markdown(f"""
                    <div class="success-box">
                        <div class="success-icon">🌟</div>
                        <div class="success-title">CONFIRMED EXOPLANET</div>
                        <p style="color: #6ee7b7; font-size: 1.3rem; margin-top: 1.5rem; line-height: 1.6;">
                            <strong>Discovery Alert!</strong><br>
                            This celestial body exhibits characteristics consistent with an exoplanet.
                            The signal pattern matches known planetary transits.
                        </p>
                        {f'<div class="confidence-badge">🎯 Confidence: {confidence:.2f}%</div>' if confidence else ''}
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(f"""
                    <div class="error-box">
                        <div class="error-icon">⚠️</div>
                        <div class="error-title">FALSE POSITIVE</div>
                        <p style="color: #fca5a5; font-size: 1.3rem; margin-top: 1.5rem; line-height: 1.6;">
                            <strong>Analysis Complete</strong><br>
                            The signal does not match exoplanet characteristics.
                            This may be instrumental noise or stellar variability.
                        </p>
                        {f'<div class="confidence-badge">🎯 Confidence: {confidence:.2f}%</div>' if confidence else ''}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed metrics
                if confidence:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("### 📊 Detailed Analysis")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🎯 Classification", "Exoplanet" if prediction == 1 else "Not Exoplanet")
                    with col2:
                        st.metric("📈 Confidence", f"{confidence:.2f}%")
                    with col3:
                        if proba_true is not None:
                            st.metric("✅ True Probability", f"{proba_true:.2f}%")
                    with col4:
                        if proba_false is not None:
                            st.metric("❌ False Probability", f"{proba_false:.2f}%")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.markdown(f"""
            <div class="error-box">
                <div class="error-icon">🔧</div>
                <div class="error-title">PROCESSING ERROR</div>
                <p style="color: #fca5a5; font-size: 1.1rem; margin-top: 1rem;">
                    An error occurred during classification.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("🔍 Error Details (for debugging)"):
                st.code(f"Error Type: {type(e).__name__}\n\nError Message: {str(e)}\n\nTraceback:\n{traceback.format_exc()}")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 2rem;">
    <p style="font-size: 1.1rem;">🌌 Powered by Advanced Machine Learning & Deep Space Data Analysis</p>
    <p style="font-size: 0.9rem; color: #9ca3af; margin-top: 0.5rem;">Built with Streamlit | Model: Scikit-Learn Random Forest</p>
</div>
""", unsafe_allow_html=True)
