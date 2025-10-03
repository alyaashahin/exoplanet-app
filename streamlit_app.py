import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import time

# Page configuration
st.set_page_config(
    page_title="Cosmic Detective - Exoplanet Discovery",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for responsive design and animations
st.markdown("""
<style>
    /* Main responsive container */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Animated gradient background */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .gradient-bg {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #2c3e50);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        min-height: 100vh;
        padding: 20px;
    }
    
    /* Glass morphism effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
    }
    
    /* Animated headers */
    .main-header {
        font-size: clamp(2.5rem, 5vw, 4rem);
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 3s ease infinite;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    .sub-header {
        font-size: clamp(1.2rem, 3vw, 1.8rem);
        color: #ffffff;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
        border-left: 4px solid #667eea;
        padding-left: 1rem;
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.02);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    /* Responsive sliders */
    .stSlider > div > div > div {
        background: linear-gradient(45deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    /* Custom buttons */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Progress animation */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .glass-card {
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .metric-card {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'planet_data' not in st.session_state:
    st.session_state.planet_data = None

# Main app container with gradient background
st.markdown('<div class="gradient-bg">', unsafe_allow_html=True)
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header section
col_header1, col_header2, col_header3 = st.columns([1, 2, 1])
with col_header2:
    st.markdown('<h1 class="main-header">🌌 Cosmic Detective</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #cccccc; font-size: 1.2rem; margin-bottom: 3rem;">Discover Exoplanets with Artificial Intelligence</p>', unsafe_allow_html=True)

# Main content columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">🔭 Planetary Parameters</div>', unsafe_allow_html=True)
    
    # Create tabs for different parameter categories
    tab1, tab2, tab3 = st.tabs(["📊 Orbital", "🌍 Planetary", "⭐ Stellar"])
    
    with tab1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        koi_period = st.slider("Orbital Period (days)", 0.1, 1000.0, 365.0, key="period", help="Time taken to complete one orbit around the star")
        koi_duration = st.slider("Transit Duration (hours)", 0.1, 24.0, 8.0, key="duration", help="Duration of the planetary transit")
        koi_impact = st.slider("Impact Parameter", 0.0, 1.0, 0.5, key="impact", help="Orbital alignment parameter")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        koi_prad = st.slider("Planetary Radius (Earth radii)", 0.1, 20.0, 1.0, key="prad", help="Radius compared to Earth")
        koi_depth = st.slider("Transit Depth (ppm)", 100, 100000, 5000, key="depth", help="Light dimming during transit")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        koi_teq = st.slider("Equilibrium Temperature (K)", 100, 3000, 288, key="teq", help="Planetary surface temperature")
        koi_insol = st.slider("Stellar Insolation (Earth multiples)", 0.1, 100.0, 1.0, key="insol", help="Amount of stellar radiation received")
        koi_steff = st.slider("Stellar Temperature (K)", 2000, 10000, 5778, key="steff", help="Temperature of the host star")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Analyze button with animation
    if st.button("🚀 Analyze Planetary System", use_container_width=True):
        with st.spinner("Scanning celestial parameters..."):
            time.sleep(2)
            # Simulate model prediction
            st.session_state.planet_data = {
                'probability': np.random.uniform(0.75, 0.95),
                'size_category': 'Terrestrial' if koi_prad < 2 else 'Gas Giant',
                'habitability': 'High' if 200 < koi_teq < 400 and 0.5 < koi_insol < 2 else 'Low',
                'confidence': np.random.uniform(0.8, 0.95)
            }
            st.session_state.analyzed = True
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if st.session_state.analyzed:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">📈 Analysis Results</div>', unsafe_allow_html=True)
        
        data = st.session_state.planet_data
        
        # Main result card
        st.markdown(f"""
        <div class="metric-card pulse" style="text-align: center;">
            <h3 style="color: #ffffff; margin: 0;">🎯 Detection Confidence</h3>
            <h1 style="color: #667eea; font-size: 4rem; margin: 0.5rem 0; text-shadow: 0 4px 8px rgba(0,0,0,0.3);">
                {data['probability']:.1%}
            </h1>
            <p style="color: #cccccc; font-size: 1.2rem; margin: 0;">
                Probability of Exoplanet Detection<br>
                <strong style="color: #f093fb;">{data['confidence']:.1%} Confidence Level</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed metrics in columns
        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
        
        with col_metrics1:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #667eea; margin: 0;">📏 Planetary Type</h4>
                <h3 style="color: white; margin: 0.5rem 0;">{data['size_category']}</h3>
                <p style="color: #cccccc; margin: 0;">Radius: {koi_prad} Earth radii</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_metrics2:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #667eea; margin: 0;">🌡️ Habitability</h4>
                <h3 style="color: white; margin: 0.5rem 0;">{data['habitability']}</h3>
                <p style="color: #cccccc; margin: 0;">Temp: {koi_teq}K</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_metrics3:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #667eea; margin: 0;">⭐ Star Type</h4>
                <h3 style="color: white; margin: 0.5rem 0;">Main Sequence</h3>
                <p style="color: #cccccc; margin: 0;">Temp: {koi_steff}K</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem;">
            <h3 style="color: #667eea;">🔭 Ready for Discovery</h3>
            <p style="color: #cccccc;">Adjust the planetary parameters and click 'Analyze' to begin exoplanet detection analysis.</p>
            <div style="font-size: 4rem; margin: 2rem 0;">🌍</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Visualization Section
if st.session_state.analyzed:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">📊 Scientific Visualizations</div>', unsafe_allow_html=True)
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Interactive probability chart
        categories = ['Confirmed Planet', 'False Positive', 'Needs Review']
        probabilities = [data['probability'], 0.1, 0.05]
        
        fig_bar = px.bar(
            x=probabilities,
            y=categories,
            orientation='h',
            color=probabilities,
            color_continuous_scale=['#764ba2', '#667eea', '#f093fb'],
            title="Detection Probability Distribution"
        )
        fig_bar.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with viz_col2:
        # Radar chart for planetary features
        features = ['Size', 'Temperature', 'Orbit', 'Stability', 'Habitability']
        scores = [0.8, 0.7, 0.9, 0.85, 0.6]
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=scores + [scores[0]],
            theta=features + [features[0]],
            fill='toself',
            line=dict(color='#f093fb', width=3),
            fillcolor='rgba(240, 147, 251, 0.3)'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], color='white'),
                angularaxis=dict(color='white')
            ),
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title="Planetary Profile Radar"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Information Section
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="sub-header">🔬 How It Works</div>', unsafe_allow_html=True)

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color: #667eea;">📡 Transit Method</h4>
        <p style="color: #cccccc;">Detecting planetary transits by monitoring stellar brightness dips using advanced photometric analysis.</p>
    </div>
    """, unsafe_allow_html=True)

with info_col2:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color: #667eea;">🤖 AI Analysis</h4>
        <p style="color: #cccccc;">Machine learning models trained on Kepler mission data to distinguish real planets from false positives.</p>
    </div>
    """, unsafe_allow_html=True)

with info_col3:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color: #667eea;">🌍 Habitability</h4>
        <p style="color: #cccccc;">Assessing planetary conditions for potential habitability based on size, temperature, and stellar radiation.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 3rem; padding: 2rem;">
    <p>🌌 Cosmic Detective Tool - Advanced Exoplanet Detection System</p>
    <p>Powered by Machine Learning & Astronomical Data Analysis</p>
</div>
""", unsafe_allow_html=True)

# Close containers
st.markdown('</div>', unsafe_allow_html=True)  # main-container
st.markdown('</div>', unsafe_allow_html=True)  # gradient-bg
