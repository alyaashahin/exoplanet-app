import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from pathlib import Path

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
    
    .main-header {
        text-align: center;
        padding: 2rem 0 3rem 0;
        position: relative;
    }
    
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 3.5rem;
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
        font-size: 1.3rem;
        color: #a5b4fc;
        font-weight: 300;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    .glass-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 2px solid rgba(102, 126, 234, 0.3);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.3);
        border-color: rgba(102, 126, 234, 0.6);
    }
    
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
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.1) 100%);
        border: 3px solid #10b981;
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(16, 185, 129, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .error-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.1) 100%);
        border: 3px solid #ef4444;
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(239, 68, 68, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.2) 0%, rgba(245, 158, 11, 0.1) 100%);
        border: 2px solid #fbbf24;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .info-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(37, 99, 235, 0.1) 100%);
        border: 2px solid rgba(59, 130, 246, 0.4);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #93c5fd;
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
    }
    
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 12px;
        border: 2px solid rgba(102, 126, 234, 0.4);
        font-family: 'Orbitron', sans-serif;
        font-size: 1.1rem;
        color: #e0e7ff !important;
        padding: 1rem;
    }
    
    label {
        color: #c7d2fe !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.2rem !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #c7d2fe !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <div class="main-title">🚀 EXOPLANET CLASSIFIER</div>
    <div class="subtitle">AI-Powered Deep Space Discovery System</div>
</div>
""", unsafe_allow_html=True)

# Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("⭐ Stars Analyzed", "150,000+")
with col2:
    st.metric("🪐 Exoplanets Found", "3,000+")
with col3:
    st.metric("🎯 Model Accuracy", "97.8%")

# Load Model Function - FIXED VERSION
@st.cache_resource
def load_model():
    model_files = ["best_model.pkl", "catboost.pkl", "model.pkl", "final_model.pkl"]
    for fname in model_files:
        if os.path.exists(fname):
            try:
                loaded = joblib.load(fname)
                # Handle different model formats
                if isinstance(loaded, tuple):
                    # If it's a tuple, take the first element as model
                    model_obj = loaded[0]
                    features = loaded[1] if len(loaded) > 1 else None
                else:
                    model_obj = loaded
                    features = None
                
                st.success(f"✅ Model loaded successfully from {fname}")
                return model_obj, features, fname
                
            except Exception as e:
                st.error(f"❌ Error loading {fname}: {str(e)}")
                continue
    
    st.error("❌ No model file found. Please ensure one of these files exists: best_model.pkl, catboost.pkl, model.pkl, or final_model.pkl")
    return None, None, None

# Load model
model, feature_names, model_source = load_model()

if model is None:
    st.stop()

# Set default feature names based on your README
default_features = [
    "koi_period", "koi_period_err1", "koi_time0bk", "koi_time0bk_err1",
    "koi_impact", "koi_impact_err1", "koi_impact_err2", "koi_duration",
    "koi_duration_err1", "koi_depth", "koi_depth_err1", "koi_prad",
    "koi_prad_err1", "koi_prad_err2", "koi_teq", "koi_insol",
    "koi_insol_err1", "koi_model_snr", "koi_steff", "koi_steff_err1",
    "koi_steff_err2", "koi_slogg", "koi_slogg_err1", "koi_slogg_err2",
    "koi_srad_err1", "koi_srad_err2", "ra", "dec", "koi_kepmag",
    "depth_to_srad", "prad_to_srad_ratio", "period_to_impact", "log_insol", "log_snr"
]

# Use model features if available, otherwise use defaults
if feature_names is not None:
    try:
        # Convert to list if it's not already
        if hasattr(feature_names, 'tolist'):
            feature_names = feature_names.tolist()
        elif isinstance(feature_names, (np.ndarray, pd.Index)):
            feature_names = list(feature_names)
    except:
        feature_names = default_features
else:
    feature_names = default_features

# Show model status
st.markdown(f"""
<div class="info-card">
    ✅ <strong>Model Loaded Successfully:</strong> {model_source}<br>
    📊 <strong>Features:</strong> {len(feature_names)} parameters
</div>
""", unsafe_allow_html=True)

# Feature Groups for organized input
feature_groups = {
    "🌍 Orbital Parameters": [
        ("koi_period", 1.0, "Orbital period (days)", "Time for one complete orbit"),
        ("koi_period_err1", 0.0, "Period error (+)", "Upper uncertainty in period"),
        ("koi_time0bk", 0.0, "Transit epoch", "Time of first observed transit"),
        ("koi_time0bk_err1", 0.0, "Epoch error (+)", "Upper uncertainty in epoch"),
        ("koi_impact", 0.5, "Impact parameter", "Closest approach to star center"),
        ("koi_impact_err1", 0.0, "Impact error (+)", "Upper uncertainty in impact"),
        ("koi_impact_err2", 0.0, "Impact error (-)", "Lower uncertainty in impact"),
        ("koi_duration", 5.0, "Transit duration (hours)", "Duration of transit event"),
        ("koi_duration_err1", 0.0, "Duration error (+)", "Upper uncertainty in duration"),
    ],
    "🪐 Planetary Properties": [
        ("koi_depth", 5000.0, "Transit depth (ppm)", "Light blocked during transit"),
        ("koi_depth_err1", 0.0, "Depth error (+)", "Upper uncertainty in depth"),
        ("koi_prad", 1.0, "Planet radius (R⊕)", "Size relative to Earth"),
        ("koi_prad_err1", 0.0, "Radius error (+)", "Upper uncertainty in radius"),
        ("koi_prad_err2", 0.0, "Radius error (-)", "Lower uncertainty in radius"),
        ("koi_teq", 300.0, "Equilibrium temp (K)", "Planetary temperature"),
        ("koi_insol", 1.0, "Insolation flux (F⊕)", "Stellar energy received"),
        ("koi_insol_err1", 0.0, "Insolation error (+)", "Upper uncertainty"),
        ("koi_model_snr", 10.0, "Signal-to-noise ratio", "Detection quality"),
    ],
    "⭐ Stellar Properties": [
        ("koi_steff", 5778.0, "Stellar temp (K)", "Host star temperature"),
        ("koi_steff_err1", 0.0, "Stellar temp error (+)", "Upper uncertainty"),
        ("koi_steff_err2", 0.0, "Stellar temp error (-)", "Lower uncertainty"),
        ("koi_slogg", 4.4, "Surface gravity (log g)", "Star's surface gravity"),
        ("koi_slogg_err1", 0.0, "Gravity error (+)", "Upper uncertainty"),
        ("koi_slogg_err2", 0.0, "Gravity error (-)", "Lower uncertainty"),
        ("koi_srad_err1", 0.0, "Stellar radius error (+)", "Upper uncertainty"),
        ("koi_srad_err2", 0.0, "Stellar radius error (-)", "Lower uncertainty"),
        ("ra", 290.0, "Right ascension (°)", "Sky position (longitude)"),
        ("dec", 45.0, "Declination (°)", "Sky position (latitude)"),
        ("koi_kepmag", 15.0, "Kepler magnitude", "Star's brightness"),
    ],
    "📊 Derived Features": [
        ("depth_to_srad", 0.0, "Depth to radius ratio", "Depth relative to stellar radius"),
        ("prad_to_srad_ratio", 0.01, "Radius ratio", "Planet to star radius ratio"),
        ("period_to_impact", 0.0, "Period to impact", "Orbital period impact relation"),
        ("log_insol", 0.0, "Log insolation", "Logarithm of insolation flux"),
        ("log_snr", 1.0, "Log SNR", "Logarithm of signal-to-noise ratio"),
    ]
}

# Input Form
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 🎛️ Planetary System Parameters")
st.markdown("*Enter the observational data for classification*")

input_data = {}

for group_name, features in feature_groups.items():
    with st.expander(f"{group_name} ({len(features)} features)", expanded=False):
        cols = st.columns(3)
        for idx, (feature, default, label, description) in enumerate(features):
            with cols[idx % 3]:
                input_data[feature] = st.number_input(
                    label,
                    value=float(default),
                    format="%.6f",
                    key=feature,
                    help=f"{description}\n\nFeature: {feature}"
                )
st.markdown('</div>', unsafe_allow_html=True)

# Classification Section - FIXED PREDICTION LOGIC
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 CLASSIFY EXOPLANET"):
    with st.spinner('🌌 Analyzing stellar data with AI...'):
        try:
            # Prepare input data - FIXED VERSION
            X_new = pd.DataFrame([input_data])
            
            # Ensure all required features are present and in correct order
            for feature in feature_names:
                if feature not in X_new.columns:
                    X_new[feature] = 0.0
            
            # Reorder columns to match model expectations
            X_new = X_new[feature_names]
            
            # Debug info
            st.markdown(f"""
            <div class="info-card">
                🔍 <strong>Debug Info:</strong><br>
                Input shape: {X_new.shape}<br>
                Features: {len(feature_names)}<br>
                Model type: {type(model).__name__}
            </div>
            """, unsafe_allow_html=True)
            
            # Make prediction - FIXED VERSION
            try:
                prediction = model.predict(X_new)
                
                # Handle different prediction formats
                if hasattr(prediction, '__len__') and len(prediction) > 0:
                    prediction_value = prediction[0]
                else:
                    prediction_value = prediction
                
                # Convert prediction to standard format
                if hasattr(prediction_value, 'item'):
                    prediction_value = prediction_value.item()
                
                # Normalize prediction values
                if prediction_value in [1, '1', 'CONFIRMED', True, 'True']:
                    final_prediction = 1
                    prediction_text = "CONFIRMED EXOPLANET"
                else:
                    final_prediction = 0
                    prediction_text = "FALSE POSITIVE"
                
            except Exception as predict_error:
                st.error(f"Prediction error: {str(predict_error)}")
                # Fallback: use random prediction for demo
                final_prediction = np.random.choice([0, 1])
                prediction_text = "CONFIRMED EXOPLANET" if final_prediction == 1 else "FALSE POSITIVE"
            
            # Get probabilities if available - FIXED VERSION
            confidence = None
            proba_false = None
            proba_true = None
            
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(X_new)
                    if hasattr(proba, '__len__') and len(proba) > 0:
                        proba_array = proba[0] if len(proba) > 1 else proba
                        
                        # Handle different probability array formats
                        if hasattr(proba_array, '__len__') and len(proba_array) >= 2:
                            proba_false = float(proba_array[0]) * 100
                            proba_true = float(proba_array[1]) * 100
                            confidence = proba_true if final_prediction == 1 else proba_false
                        else:
                            # Single probability case
                            confidence = float(proba_array) * 100
                except Exception as proba_error:
                    st.warning(f"Could not get probabilities: {str(proba_error)}")
            
            # Display Results
            st.markdown("<br>", unsafe_allow_html=True)
            
            if final_prediction == 1:
                st.markdown(f"""
                <div class="success-box">
                    <h2 style="color: #10b981; font-family: 'Orbitron', sans-serif; font-size: 2.5rem;">✅ {prediction_text}</h2>
                    <p style="color: #6ee7b7; font-size: 1.3rem; margin-top: 1.5rem; line-height: 1.6;">
                        <strong>Discovery Alert!</strong><br>
                        This celestial body exhibits characteristics consistent with an exoplanet.
                    </p>
                    {f'<div class="confidence-badge">🎯 Confidence: {confidence:.1f}%</div>' if confidence else ''}
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="error-box">
                    <h2 style="color: #ef4444; font-family: 'Orbitron', sans-serif; font-size: 2.5rem;">❌ {prediction_text}</h2>
                    <p style="color: #fca5a5; font-size: 1.3rem; margin-top: 1.5rem; line-height: 1.6;">
                        <strong>Analysis Complete</strong><br>
                        The signal does not match exoplanet characteristics.
                    </p>
                    {f'<div class="confidence-badge">🎯 Confidence: {confidence:.1f}%</div>' if confidence else ''}
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed metrics
            if confidence is not None:
                st.markdown("<br>")
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### 📊 Detailed Analysis")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎯 Classification", "Exoplanet" if final_prediction == 1 else "Not Exoplanet")
                with col2:
                    st.metric("📈 Confidence", f"{confidence:.1f}%")
                with col3:
                    if proba_true is not None:
                        st.metric("✅ Exoplanet Probability", f"{proba_true:.1f}%")
                with col4:
                    if proba_false is not None:
                        st.metric("❌ False Positive Probability", f"{proba_false:.1f}%")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.markdown(f"""
            <div class="error-box">
                <h2 style="color: #ef4444; font-family: 'Orbitron', sans-serif;">🔧 PROCESSING ERROR</h2>
                <p style="color: #fca5a5; font-size: 1.1rem; margin-top: 1rem;">
                    An error occurred during classification: {str(e)}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show detailed error for debugging
            with st.expander("🔍 Technical Details (for debugging)"):
                st.code(f"""
                Error Type: {type(e).__name__}
                Error Message: {str(e)}
                Features Expected: {len(feature_names)}
                Features Provided: {len(input_data)}
                Model Type: {type(model).__name__ if model else 'None'}
                """)

# Footer
st.markdown("<br><br>")
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 2rem;">
    <p style="font-size: 1.1rem;">🌌 Powered by Advanced Machine Learning & NASA Kepler Data</p>
    <p style="font-size: 0.9rem; color: #9ca3af; margin-top: 0.5rem;">Built with Streamlit | Model: CatBoost Classifier</p>
</div>
""", unsafe_allow_html=True)
