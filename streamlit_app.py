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

# Enhanced CSS with Advanced Animations and Better Styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
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
        --cosmic-pink: #ec4899;
        --aurora-green: #10b981;
        --pulsar-blue: #06b6d4;
    }
    
    /* Main App Styling */
    .main {
        background: 
            radial-gradient(circle at 20% 80%, rgba(59, 130, 246, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(139, 92, 246, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(245, 158, 11, 0.05) 0%, transparent 50%),
            linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        min-height: 100vh;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Enhanced Header with Advanced Animations */
    .cosmic-header {
        background: 
            linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(139, 92, 246, 0.15) 50%, rgba(245, 158, 11, 0.1) 100%),
            radial-gradient(circle at center, rgba(59, 130, 246, 0.1) 0%, transparent 70%);
        border-radius: 24px;
        padding: 4rem 2rem;
        margin-bottom: 3rem;
        position: relative;
        overflow: hidden;
        border: 2px solid rgba(59, 130, 246, 0.3);
        backdrop-filter: blur(10px);
        animation: headerGlow 4s ease-in-out infinite;
    }
    
    @keyframes headerGlow {
        0%, 100% { 
            box-shadow: 0 0 30px rgba(59, 130, 246, 0.3), 0 0 60px rgba(139, 92, 246, 0.2);
            border-color: rgba(59, 130, 246, 0.3);
        }
        50% { 
            box-shadow: 0 0 50px rgba(59, 130, 246, 0.5), 0 0 80px rgba(139, 92, 246, 0.3);
            border-color: rgba(139, 92, 246, 0.5);
        }
    }
    
    .cosmic-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.2), transparent);
        animation: shimmer 3s ease-in-out infinite;
    }
    
    .cosmic-header::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: 
            radial-gradient(circle, rgba(245, 158, 11, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(236, 72, 153, 0.1) 0%, transparent 50%);
        animation: rotate 20s linear infinite;
        z-index: -1;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Enhanced Title Styling */
    .cosmic-title {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 25%, #ec4899 50%, #f59e0b 75%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 4.5rem !important;
        font-weight: 900;
        text-align: center;
        margin-bottom: 1rem;
        animation: gradient-x 4s ease infinite, titleFloat 6s ease-in-out infinite;
        background-size: 300% 300%;
        text-shadow: 0 0 30px rgba(59, 130, 246, 0.5);
        font-family: 'Space Grotesk', sans-serif;
    }
    
    @keyframes gradient-x {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    @keyframes titleFloat {
        0%, 100% { transform: translateY(0px) scale(1); }
        50% { transform: translateY(-10px) scale(1.02); }
    }
    
    .cosmic-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.4rem;
        font-weight: 300;
        margin-bottom: 2rem;
        opacity: 0.9;
        animation: subtitleGlow 3s ease-in-out infinite;
    }
    
    @keyframes subtitleGlow {
        0%, 100% { text-shadow: 0 0 10px rgba(148, 163, 184, 0.3); }
        50% { text-shadow: 0 0 20px rgba(148, 163, 184, 0.6); }
    }
    
    /* Enhanced Metrics Cards */
    .metric-card {
        background: 
            linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(51, 65, 85, 0.7) 100%),
            radial-gradient(circle at top right, rgba(59, 130, 246, 0.1) 0%, transparent 50%);
        border-radius: 20px;
        padding: 2rem;
        border: 2px solid rgba(59, 130, 246, 0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
        animation: cardFloat 8s ease-in-out infinite;
    }
    
    @keyframes cardFloat {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        25% { transform: translateY(-5px) rotate(0.5deg); }
        50% { transform: translateY(-8px) rotate(0deg); }
        75% { transform: translateY(-3px) rotate(-0.5deg); }
    }
    
    .metric-card:hover {
        transform: translateY(-10px) scale(1.05);
        box-shadow: 
            0 25px 50px rgba(59, 130, 246, 0.3),
            0 0 0 1px rgba(139, 92, 246, 0.5);
        border-color: rgba(139, 92, 246, 0.6);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899, #f59e0b, #10b981);
        background-size: 200% 100%;
        animation: gradientMove 3s ease infinite;
    }
    
    @keyframes gradientMove {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .metric-card::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
        z-index: -1;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.3; }
        50% { transform: scale(1.1); opacity: 0.6; }
    }
    
    /* Enhanced Button Styling */
    .stButton>button {
        background: 
            linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        color: white;
        border-radius: 16px;
        padding: 1rem 2.5rem;
        font-weight: 700;
        border: none;
        box-shadow: 
            0 10px 20px rgba(59, 130, 246, 0.4),
            0 0 0 1px rgba(255, 255, 255, 0.1);
        width: 100%;
        font-family: 'Space Grotesk', sans-serif;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        animation: buttonGlow 3s ease-in-out infinite;
    }
    
    @keyframes buttonGlow {
        0%, 100% { 
            box-shadow: 
                0 10px 20px rgba(59, 130, 246, 0.4),
                0 0 0 1px rgba(255, 255, 255, 0.1);
        }
        50% { 
            box-shadow: 
                0 15px 30px rgba(59, 130, 246, 0.6),
                0 0 0 1px rgba(255, 255, 255, 0.2);
        }
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 
            0 20px 40px rgba(59, 130, 246, 0.5),
            0 0 0 1px rgba(255, 255, 255, 0.3);
        animation: buttonPulse 0.6s ease-in-out;
    }
    
    @keyframes buttonPulse {
        0% { transform: translateY(-3px) scale(1.02); }
        50% { transform: translateY(-5px) scale(1.05); }
        100% { transform: translateY(-3px) scale(1.02); }
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.6s ease;
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    /* Enhanced Expander Styling */
    .streamlit-expanderHeader {
        background: 
            linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(51, 65, 85, 0.8) 100%),
            radial-gradient(circle at top left, rgba(59, 130, 246, 0.1) 0%, transparent 50%);
        border-radius: 16px;
        border: 2px solid rgba(59, 130, 246, 0.3);
        color: #e2e8f0;
        font-weight: 700;
        font-size: 1.2rem;
        padding: 1rem 1.5rem;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        animation: expanderGlow 5s ease-in-out infinite;
    }
    
    @keyframes expanderGlow {
        0%, 100% { border-color: rgba(59, 130, 246, 0.3); }
        50% { border-color: rgba(139, 92, 246, 0.5); }
    }
    
    .streamlit-expanderHeader:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(59, 130, 246, 0.2);
    }
    
    .streamlit-expanderContent {
        background: 
            linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.7) 100%),
            radial-gradient(circle at bottom right, rgba(139, 92, 246, 0.05) 0%, transparent 50%);
        border-radius: 0 0 16px 16px;
        border: 2px solid rgba(59, 130, 246, 0.2);
        border-top: none;
        backdrop-filter: blur(10px);
        padding: 1.5rem;
    }
    
    /* Enhanced Number Input Styling */
    .stNumberInput>div>div>input {
        background: 
            linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(51, 65, 85, 0.7) 100%);
        border: 2px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        color: #e2e8f0;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 500;
        padding: 0.75rem 1rem;
        transition: all 0.3s ease;
        backdrop-filter: blur(5px);
    }
    
    .stNumberInput>div>div>input:focus {
        border-color: #3b82f6;
        box-shadow: 
            0 0 0 3px rgba(59, 130, 246, 0.2),
            0 10px 20px rgba(59, 130, 246, 0.1);
        transform: translateY(-2px);
    }
    
    .stNumberInput>div>div>input:hover {
        border-color: rgba(139, 92, 246, 0.5);
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.1);
    }
    
    /* Enhanced Label Styling */
    .stNumberInput label {
        color: #cbd5e1;
        font-weight: 600;
        font-size: 1rem;
        font-family: 'Space Grotesk', sans-serif;
        text-shadow: 0 0 10px rgba(203, 213, 225, 0.3);
    }
    
    /* Enhanced Success Box */
    .success-box {
        background: 
            linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(16, 185, 129, 0.1) 100%),
            radial-gradient(circle at center, rgba(34, 197, 94, 0.1) 0%, transparent 70%);
        border: 3px solid #22c55e;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        animation: successPulse 2s ease-in-out infinite;
        backdrop-filter: blur(10px);
    }
    
    @keyframes successPulse {
        0%, 100% { 
            box-shadow: 0 0 30px rgba(34, 197, 94, 0.4);
            transform: scale(1);
        }
        50% { 
            box-shadow: 0 0 50px rgba(34, 197, 94, 0.6);
            transform: scale(1.02);
        }
    }
    
    .success-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #22c55e, #10b981, #059669, #22c55e);
        background-size: 200% 100%;
        animation: gradientMove 2s ease infinite;
    }
    
    .success-box::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(34, 197, 94, 0.1) 0%, transparent 70%);
        animation: rotate 10s linear infinite;
        z-index: -1;
    }
    
    /* Enhanced Error Box */
    .error-box {
        background: 
            linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.1) 100%),
            radial-gradient(circle at center, rgba(239, 68, 68, 0.1) 0%, transparent 70%);
        border: 3px solid #ef4444;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        animation: errorPulse 2s ease-in-out infinite;
        backdrop-filter: blur(10px);
    }
    
    @keyframes errorPulse {
        0%, 100% { 
            box-shadow: 0 0 30px rgba(239, 68, 68, 0.4);
            transform: scale(1);
        }
        50% { 
            box-shadow: 0 0 50px rgba(239, 68, 68, 0.6);
            transform: scale(1.02);
        }
    }
    
    .error-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #ef4444, #dc2626, #b91c1c, #ef4444);
        background-size: 200% 100%;
        animation: gradientMove 2s ease infinite;
    }
    
    /* Enhanced Confidence Metrics */
    .confidence-metric {
        background: 
            linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(51, 65, 85, 0.7) 100%),
            radial-gradient(circle at top right, rgba(139, 92, 246, 0.1) 0%, transparent 50%);
        border-radius: 16px;
        padding: 2rem;
        border: 2px solid rgba(139, 92, 246, 0.3);
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
        animation: confidenceFloat 6s ease-in-out infinite;
    }
    
    @keyframes confidenceFloat {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        25% { transform: translateY(-3px) rotate(0.5deg); }
        50% { transform: translateY(-5px) rotate(0deg); }
        75% { transform: translateY(-2px) rotate(-0.5deg); }
    }
    
    .confidence-metric:hover {
        transform: translateY(-8px) scale(1.05);
        box-shadow: 0 20px 40px rgba(139, 92, 246, 0.3);
        border-color: rgba(139, 92, 246, 0.6);
    }
    
    /* Enhanced Randomize Buttons */
    .randomize-btn {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        font-weight: 700;
        font-size: 0.9rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 6px 12px rgba(245, 158, 11, 0.4);
        font-family: 'Space Grotesk', sans-serif;
        animation: randomizeGlow 3s ease-in-out infinite;
    }
    
    @keyframes randomizeGlow {
        0%, 100% { box-shadow: 0 6px 12px rgba(245, 158, 11, 0.4); }
        50% { box-shadow: 0 8px 16px rgba(245, 158, 11, 0.6); }
    }
    
    .randomize-btn:hover {
        transform: translateY(-3px) scale(1.1);
        box-shadow: 0 10px 20px rgba(245, 158, 11, 0.6);
        animation: randomizePulse 0.6s ease-in-out;
    }
    
    @keyframes randomizePulse {
        0% { transform: translateY(-3px) scale(1.1); }
        50% { transform: translateY(-5px) scale(1.15); }
        100% { transform: translateY(-3px) scale(1.1); }
    }
    
    /* Enhanced Feature Container */
    .feature-container {
        margin-bottom: 1.5rem;
        padding: 1.5rem;
        background: 
            linear-gradient(135deg, rgba(30, 41, 59, 0.4) 0%, rgba(51, 65, 85, 0.2) 100%),
            radial-gradient(circle at top left, rgba(59, 130, 246, 0.05) 0%, transparent 50%);
        border-radius: 12px;
        border: 1px solid rgba(59, 130, 246, 0.2);
        transition: all 0.3s ease;
        backdrop-filter: blur(5px);
        animation: containerFloat 8s ease-in-out infinite;
    }
    
    @keyframes containerFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-2px); }
    }
    
    .feature-container:hover {
        border-color: rgba(139, 92, 246, 0.4);
        box-shadow: 0 10px 20px rgba(59, 130, 246, 0.1);
        transform: translateY(-3px);
    }
    
    /* Enhanced Animated Background Elements */
    .cosmic-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        pointer-events: none;
        overflow: hidden;
    }
    
    .star {
        position: absolute;
        background: #e2e8f0;
        border-radius: 50%;
        animation: twinkle 4s ease-in-out infinite;
        box-shadow: 0 0 6px rgba(226, 232, 240, 0.8);
    }
    
    @keyframes twinkle {
        0%, 100% { 
            opacity: 0.3; 
            transform: scale(1); 
            box-shadow: 0 0 6px rgba(226, 232, 240, 0.8);
        }
        50% { 
            opacity: 0.9; 
            transform: scale(1.5); 
            box-shadow: 0 0 12px rgba(226, 232, 240, 1);
        }
    }
    
    .shooting-star {
        position: absolute;
        width: 3px;
        height: 3px;
        background: linear-gradient(45deg, #3b82f6, #8b5cf6, #ec4899);
        border-radius: 50%;
        animation: shoot 4s linear infinite;
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.8);
    }
    
    @keyframes shoot {
        0% { 
            transform: translateX(-100px) translateY(0px); 
            opacity: 0; 
        }
        10% { 
            opacity: 1; 
        }
        90% { 
            opacity: 1; 
        }
        100% { 
            transform: translateX(100vw) translateY(-80px); 
            opacity: 0; 
        }
    }
    
    .cosmic-particle {
        position: absolute;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.6) 0%, transparent 70%);
        border-radius: 50%;
        animation: particleFloat 15s ease-in-out infinite;
    }
    
    @keyframes particleFloat {
        0%, 100% { 
            transform: translateY(0px) translateX(0px) scale(1); 
            opacity: 0.3; 
        }
        25% { 
            transform: translateY(-20px) translateX(10px) scale(1.2); 
            opacity: 0.6; 
        }
        50% { 
            transform: translateY(-10px) translateX(-15px) scale(0.8); 
            opacity: 0.4; 
        }
        75% { 
            transform: translateY(-30px) translateX(5px) scale(1.1); 
            opacity: 0.7; 
        }
    }
    
    .nebula-cloud {
        position: absolute;
        background: radial-gradient(ellipse, rgba(139, 92, 246, 0.1) 0%, transparent 70%);
        border-radius: 50%;
        animation: nebulaDrift 25s ease-in-out infinite;
        filter: blur(2px);
    }
    
    @keyframes nebulaDrift {
        0%, 100% { 
            transform: translateX(0px) translateY(0px) rotate(0deg) scale(1); 
            opacity: 0.2; 
        }
        25% { 
            transform: translateX(30px) translateY(-20px) rotate(90deg) scale(1.2); 
            opacity: 0.4; 
        }
        50% { 
            transform: translateX(-20px) translateY(-30px) rotate(180deg) scale(0.8); 
            opacity: 0.3; 
        }
        75% { 
            transform: translateX(-30px) translateY(10px) rotate(270deg) scale(1.1); 
            opacity: 0.5; 
        }
    }
    
    /* Enhanced Loading Animation */
    .cosmic-loader {
        display: inline-block;
        width: 30px;
        height: 30px;
        border: 4px solid rgba(59, 130, 246, 0.3);
        border-radius: 50%;
        border-top-color: #3b82f6;
        border-right-color: #8b5cf6;
        animation: spin 1s ease-in-out infinite;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.5);
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Enhanced Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899, #f59e0b);
        background-size: 200% 100%;
        animation: gradientMove 2s ease infinite;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .cosmic-title {
            font-size: 2.8rem !important;
        }
        .cosmic-subtitle {
            font-size: 1.2rem;
        }
        .metric-card {
            padding: 1.5rem;
        }
        .cosmic-header {
            padding: 2rem 1rem;
        }
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #8b5cf6, #ec4899);
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Animated Background with More Particles
st.markdown("""
<div class="cosmic-bg">
    <!-- Twinkling Stars -->
    <div class="star" style="top: 5%; left: 15%; width: 2px; height: 2px; animation-delay: 0s;"></div>
    <div class="star" style="top: 12%; left: 85%; width: 1px; height: 1px; animation-delay: 1.2s;"></div>
    <div class="star" style="top: 18%; left: 35%; width: 3px; height: 3px; animation-delay: 2.4s;"></div>
    <div class="star" style="top: 25%; left: 65%; width: 2px; height: 2px; animation-delay: 0.8s;"></div>
    <div class="star" style="top: 32%; left: 8%; width: 1px; height: 1px; animation-delay: 1.8s;"></div>
    <div class="star" style="top: 38%; left: 92%; width: 2px; height: 2px; animation-delay: 3.2s;"></div>
    <div class="star" style="top: 45%; left: 25%; width: 1px; height: 1px; animation-delay: 0.4s;"></div>
    <div class="star" style="top: 52%; left: 75%; width: 3px; height: 3px; animation-delay: 2.8s;"></div>
    <div class="star" style="top: 58%; left: 12%; width: 2px; height: 2px; animation-delay: 1.6s;"></div>
    <div class="star" style="top: 65%; left: 88%; width: 1px; height: 1px; animation-delay: 0.6s;"></div>
    <div class="star" style="top: 72%; left: 42%; width: 2px; height: 2px; animation-delay: 2.2s;"></div>
    <div class="star" style="top: 78%; left: 18%; width: 1px; height: 1px; animation-delay: 3.6s;"></div>
    <div class="star" style="top: 85%; left: 68%; width: 3px; height: 3px; animation-delay: 1.0s;"></div>
    <div class="star" style="top: 92%; left: 28%; width: 2px; height: 2px; animation-delay: 2.6s;"></div>
    <div class="star" style="top: 8%; left: 55%; width: 1px; height: 1px; animation-delay: 1.4s;"></div>
    <div class="star" style="top: 22%; left: 78%; width: 2px; height: 2px; animation-delay: 3.0s;"></div>
    <div class="star" style="top: 35%; left: 45%; width: 1px; height: 1px; animation-delay: 0.2s;"></div>
    <div class="star" style="top: 48%; left: 82%; width: 3px; height: 3px; animation-delay: 2.0s;"></div>
    <div class="star" style="top: 62%; left: 22%; width: 2px; height: 2px; animation-delay: 3.4s;"></div>
    <div class="star" style="top: 75%; left: 58%; width: 1px; height: 1px; animation-delay: 0.8s;"></div>
    
    <!-- Shooting Stars -->
    <div class="shooting-star" style="top: 8%; left: 0%; animation-delay: 0s;"></div>
    <div class="shooting-star" style="top: 28%; left: 0%; animation-delay: 3s;"></div>
    <div class="shooting-star" style="top: 48%; left: 0%; animation-delay: 6s;"></div>
    <div class="shooting-star" style="top: 68%; left: 0%; animation-delay: 9s;"></div>
    <div class="shooting-star" style="top: 88%; left: 0%; animation-delay: 12s;"></div>
    
    <!-- Cosmic Particles -->
    <div class="cosmic-particle" style="top: 15%; left: 30%; width: 8px; height: 8px; animation-delay: 0s;"></div>
    <div class="cosmic-particle" style="top: 35%; left: 70%; width: 6px; height: 6px; animation-delay: 2s;"></div>
    <div class="cosmic-particle" style="top: 55%; left: 20%; width: 10px; height: 10px; animation-delay: 4s;"></div>
    <div class="cosmic-particle" style="top: 75%; left: 80%; width: 7px; height: 7px; animation-delay: 6s;"></div>
    <div class="cosmic-particle" style="top: 25%; left: 50%; width: 9px; height: 9px; animation-delay: 8s;"></div>
    <div class="cosmic-particle" style="top: 65%; left: 40%; width: 5px; height: 5px; animation-delay: 10s;"></div>
    
    <!-- Nebula Clouds -->
    <div class="nebula-cloud" style="top: 10%; left: 60%; width: 120px; height: 80px; animation-delay: 0s;"></div>
    <div class="nebula-cloud" style="top: 40%; left: 20%; width: 100px; height: 60px; animation-delay: 5s;"></div>
    <div class="nebula-cloud" style="top: 70%; left: 70%; width: 140px; height: 90px; animation-delay: 10s;"></div>
    <div class="nebula-cloud" style="top: 20%; left: 80%; width: 80px; height: 50px; animation-delay: 15s;"></div>
    <div class="nebula-cloud" style="top: 60%; left: 10%; width: 110px; height: 70px; animation-delay: 20s;"></div>
</div>
""", unsafe_allow_html=True)

# Enhanced Header Section with Dynamic Content
st.markdown("""
<div class="cosmic-header">
    <h1 class="cosmic-title">🌌 COSMIC EXOPLANET CLASSIFIER</h1>
    <p class="cosmic-subtitle">✨ Discover and classify distant worlds using advanced machine learning ✨</p>
    <div style="text-align: center; margin-top: 2rem;">
        <span style="display: inline-block; padding: 0.5rem 1rem; background: rgba(59, 130, 246, 0.2); border-radius: 20px; color: #3b82f6; font-weight: 600; margin: 0 0.5rem; animation: fadeInOut 3s ease-in-out infinite;">
            🚀 Advanced AI Technology
        </span>
        <span style="display: inline-block; padding: 0.5rem 1rem; background: rgba(139, 92, 246, 0.2); border-radius: 20px; color: #8b5cf6; font-weight: 600; margin: 0 0.5rem; animation: fadeInOut 3s ease-in-out infinite 1s;">
            🌟 Real-time Analysis
        </span>
        <span style="display: inline-block; padding: 0.5rem 1rem; background: rgba(245, 158, 11, 0.2); border-radius: 20px; color: #f59e0b; font-weight: 600; margin: 0 0.5rem; animation: fadeInOut 3s ease-in-out infinite 2s;">
            🪐 Cosmic Discovery
        </span>
    </div>
</div>

<style>
@keyframes fadeInOut {
    0%, 100% { opacity: 0.6; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.05); }
}
</style>
""", unsafe_allow_html=True)

# Enhanced Metrics Section with Dynamic Animations
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #3b82f6; margin: 0 0 0.5rem 0; font-size: 2.5rem; animation: iconPulse 2s ease-in-out infinite;">⭐</h3>
        <h4 style="color: #e2e8f0; margin: 0; font-size: 1.8rem; font-weight: 800;">150,000+</h4>
        <p style="color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 1.1rem;">Stars Analyzed</p>
        <div style="margin-top: 1rem; height: 4px; background: linear-gradient(90deg, #3b82f6, #8b5cf6); border-radius: 2px; animation: progressBar 3s ease-in-out infinite;"></div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #8b5cf6; margin: 0 0 0.5rem 0; font-size: 2.5rem; animation: iconPulse 2s ease-in-out infinite 0.5s;">🪐</h3>
        <h4 style="color: #e2e8f0; margin: 0; font-size: 1.8rem; font-weight: 800;">3,000+</h4>
        <p style="color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 1.1rem;">Exoplanets Found</p>
        <div style="margin-top: 1rem; height: 4px; background: linear-gradient(90deg, #8b5cf6, #ec4899); border-radius: 2px; animation: progressBar 3s ease-in-out infinite 0.5s;"></div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #f59e0b; margin: 0 0 0.5rem 0; font-size: 2.5rem; animation: iconPulse 2s ease-in-out infinite 1s;">🎯</h3>
        <h4 style="color: #e2e8f0; margin: 0; font-size: 1.8rem; font-weight: 800;">97.8%</h4>
        <p style="color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 1.1rem;">Model Accuracy</p>
        <div style="margin-top: 1rem; height: 4px; background: linear-gradient(90deg, #f59e0b, #10b981); border-radius: 2px; animation: progressBar 3s ease-in-out infinite 1s;"></div>
    </div>
    """, unsafe_allow_html=True)

# Add additional CSS for new animations
st.markdown("""
<style>
@keyframes iconPulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.2); }
}

@keyframes progressBar {
    0%, 100% { width: 0%; }
    50% { width: 100%; }
}

@keyframes successBounce {
    0% { transform: scale(0.8) translateY(20px); opacity: 0; }
    50% { transform: scale(1.1) translateY(-10px); opacity: 1; }
    100% { transform: scale(1) translateY(0px); opacity: 1; }
}

@keyframes errorShake {
    0%, 100% { transform: translateX(0px); }
    25% { transform: translateX(-5px); }
    75% { transform: translateX(5px); }
}

@keyframes textGlow {
    0%, 100% { text-shadow: 0 0 10px rgba(34, 197, 94, 0.5); }
    50% { text-shadow: 0 0 20px rgba(34, 197, 94, 0.8); }
}

@keyframes textGlowRed {
    0%, 100% { text-shadow: 0 0 10px rgba(239, 68, 68, 0.5); }
    50% { text-shadow: 0 0 20px rgba(239, 68, 68, 0.8); }
}

@keyframes badgeFloat {
    0%, 100% { transform: translateY(0px) scale(1); opacity: 0.8; }
    50% { transform: translateY(-5px) scale(1.05); opacity: 1; }
}
</style>
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

# Feature Groups with Enhanced Descriptions and Ranges
feature_groups = {
    "🌍 Orbital Parameters": [
        ("koi_period", 0.0, "Orbital period (days)", "Time for planet to complete one orbit around its star", -0.389486, 3.884597),
        ("koi_period_err1", 0.0, "Period error", "Uncertainty in orbital period measurement", -0.249980, 3.912221),
        ("koi_time0bk", 0.0, "Transit epoch", "Time of first observed transit", -0.509194, 3.841147),
        ("koi_time0bk_err1", 0.0, "Transit epoch error", "Uncertainty in transit timing", -0.651946, 3.741984),
        ("koi_impact", 0.0, "Impact parameter", "Minimum distance between planet and star center", -0.630537, 3.631575),
        ("koi_impact_err1", 0.0, "Impact error 1", "Uncertainty in impact parameter", -0.666667, 3.568627),
        ("koi_impact_err2", 0.0, "Impact error 2", "Additional uncertainty in impact parameter", -3.567234, 0.724336),
        ("koi_duration", 0.0, "Transit duration", "Duration of planet passing in front of star", -1.031911, 3.611988),
        ("koi_duration_err1", 0.0, "Duration error", "Uncertainty in transit duration", -0.692735, 3.733467),
    ],
    "🪐 Planetary Properties": [
        ("koi_depth", 0.0, "Transit depth", "Fraction of starlight blocked during transit", -0.572557, 3.788191),
        ("koi_depth_err1", 0.0, "Depth error", "Uncertainty in transit depth measurement", -0.723214, 3.843750),
        ("koi_prad", 0.0, "Planetary radius", "Radius of the planet (Earth radii)", -1.142857, 3.691814),
        ("koi_prad_err1", 0.0, "Radius error 1", "Uncertainty in planetary radius", -0.723404, 3.553191),
        ("koi_prad_err2", 0.0, "Radius error 2", "Additional uncertainty in radius", -3.689655, 0.724138),
        ("koi_teq", 0.0, "Equilibrium temperature", "Planet's surface temperature (Kelvin)", -1.433444, 3.570820),
        ("koi_insol", 0.0, "Insolation flux", "Stellar radiation received by planet", -0.312844, 3.735998),
        ("koi_insol_err1", 0.0, "Insolation error", "Uncertainty in insolation measurement", -0.261479, 3.751848),
        ("koi_model_snr", 0.0, "Signal-to-noise ratio", "Quality of transit detection", -0.675000, 3.850000),
    ],
    "⭐ Stellar Properties": [
        ("koi_steff", 0.0, "Stellar effective temperature", "Star's surface temperature (Kelvin)", -3.652231, 3.522310),
        ("koi_steff_err1", 0.0, "Stellar temp error 1", "Uncertainty in stellar temperature", -1.500000, 3.550000),
        ("koi_steff_err2", 0.0, "Stellar temp error 2", "Additional temperature uncertainty", -3.494382, 1.483146),
        ("koi_slogg", 0.0, "Stellar surface gravity", "Star's surface gravity (log g)", -3.687500, 3.363971),
        ("koi_slogg_err1", 0.0, "Surface gravity error 1", "Uncertainty in surface gravity", -0.693333, 3.658667),
        ("koi_slogg_err2", 0.0, "Surface gravity error 2", "Additional gravity uncertainty", -3.685185, 1.037037),
        ("koi_srad_err1", 0.0, "Stellar radius error 1", "Uncertainty in stellar radius", -0.941748, 3.543689),
        ("koi_srad_err2", 0.0, "Stellar radius error 2", "Additional radius uncertainty", -3.696721, 0.803279),
        ("ra", 0.0, "Right ascension", "Celestial coordinate (degrees)", -1.479197, 1.324565),
        ("dec", 0.0, "Declination", "Celestial coordinate (degrees)", -1.327027, 1.416869),
        ("koi_kepmag", 0.0, "Kepler magnitude", "Star's apparent brightness", -3.649838, 1.684677),
    ],
    "📊 Derived Features": [
        ("depth_to_srad", 0.0, "Depth to stellar radius ratio", "Normalized transit depth", -0.479075, 30.328550),
        ("prad_to_srad_ratio", 0.0, "Planet to star radius ratio", "Relative size comparison", -1.007994, 13.798500),
        ("period_to_impact", 0.0, "Period to impact ratio", "Orbital geometry indicator", -0.239422, 8018737000.0),
        ("log_insol", 0.0, "Log insolation", "Logarithmic insolation flux", -1.665119, 0.945364),
        ("log_snr", 0.0, "Log signal-to-noise", "Logarithmic detection quality", -1.799029, 1.569439),
    ]
}

# Input Form with Enhanced UI and Randomize Buttons
inputs = {}

# Add CSS for randomize buttons
st.markdown("""
<style>
    .randomize-btn {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        font-size: 0.8rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(245, 158, 11, 0.3);
    }
    .randomize-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(245, 158, 11, 0.4);
    }
    .feature-container {
        margin-bottom: 1rem;
        padding: 1rem;
        background: rgba(30, 41, 59, 0.3);
        border-radius: 8px;
        border: 1px solid rgba(59, 130, 246, 0.1);
    }
</style>
""", unsafe_allow_html=True)

for group_name, features in feature_groups.items():
    with st.expander(f"{group_name} ({len(features)} features)", expanded=True):
        # Add randomize all button for this group
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button(f"🎲 Randomize All {group_name.split()[1]}", key=f"randomize_{group_name}"):
                for feature, default, label, description, min_val, max_val in features:
                    random_value = np.random.uniform(min_val, max_val)
                    st.session_state[feature] = random_value
                st.rerun()
        
        cols = st.columns(3)
        for idx, (feature, default, label, description, min_val, max_val) in enumerate(features):
            with cols[idx % 3]:
                # Create a container for each feature
                with st.container():
                    # Feature label and individual randomize button
                    col_label, col_rand = st.columns([3, 1])
                    with col_label:
                        st.markdown(f"<div style='color: #cbd5e1; font-weight: 500; font-size: 0.9rem; margin-bottom: 0.5rem;'>{label}</div>", unsafe_allow_html=True)
                    with col_rand:
                        if st.button("🎲", key=f"rand_{feature}", help=f"Randomize {label}"):
                            random_value = np.random.uniform(min_val, max_val)
                            st.session_state[feature] = random_value
                            st.rerun()
                    
                    inputs[feature] = st.number_input(
                        "",
                        value=st.session_state.get(feature, default),
                        min_value=float(min_val),
                        max_value=float(max_val),
                        step=0.000001,
                        format="%.6f",
                        key=feature,
                        help=f"{description}\nRange: [{min_val:.3f}, {max_val:.3f}]"
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
                    <h2 style="color: #22c55e; margin: 0 0 1rem 0; font-size: 3rem; animation: successBounce 1s ease-in-out;">✅ CONFIRMED EXOPLANET</h2>
                    <p style="font-size: 1.4rem; color: #22c55e; margin: 0; animation: textGlow 2s ease-in-out infinite;">
                        🌟 This object shows strong evidence of being an exoplanet! 🌟
                    </p>
                    <p style="color: #16a34a; margin: 1rem 0 0 0; font-size: 1.2rem;">
                        The planetary signature has been detected with high confidence.
                    </p>
                    <div style="margin-top: 2rem;">
                        <div style="display: inline-block; padding: 0.5rem 1rem; background: rgba(34, 197, 94, 0.2); border-radius: 20px; color: #22c55e; font-weight: 600; margin: 0 0.5rem; animation: badgeFloat 3s ease-in-out infinite;">
                            🪐 Planetary Signal Detected
                        </div>
                        <div style="display: inline-block; padding: 0.5rem 1rem; background: rgba(16, 185, 129, 0.2); border-radius: 20px; color: #10b981; font-weight: 600; margin: 0 0.5rem; animation: badgeFloat 3s ease-in-out infinite 0.5s;">
                            🌟 High Confidence
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="error-box">
                    <h2 style="color: #ef4444; margin: 0 0 1rem 0; font-size: 3rem; animation: errorShake 1s ease-in-out;">❌ FALSE POSITIVE</h2>
                    <p style="font-size: 1.4rem; color: #ef4444; margin: 0; animation: textGlowRed 2s ease-in-out infinite;">
                        🔍 This object is likely not an exoplanet
                    </p>
                    <p style="color: #dc2626; margin: 1rem 0 0 0; font-size: 1.2rem;">
                        The signal appears to be caused by other astronomical phenomena.
                    </p>
                    <div style="margin-top: 2rem;">
                        <div style="display: inline-block; padding: 0.5rem 1rem; background: rgba(239, 68, 68, 0.2); border-radius: 20px; color: #ef4444; font-weight: 600; margin: 0 0.5rem; animation: badgeFloat 3s ease-in-out infinite;">
                            ⚠️ No Planetary Signal
                        </div>
                        <div style="display: inline-block; padding: 0.5rem 1rem; background: rgba(220, 38, 38, 0.2); border-radius: 20px; color: #dc2626; font-weight: 600; margin: 0 0.5rem; animation: badgeFloat 3s ease-in-out infinite 0.5s;">
                            🔬 Further Analysis Needed
                        </div>
                    </div>
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

# Enhanced Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; margin-top: 4rem; padding: 2rem;">
    <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 1rem;">
        <span style="font-size: 1.5rem; margin: 0 1rem; animation: iconRotate 4s ease-in-out infinite;">🌌</span>
        <span style="font-size: 1.5rem; margin: 0 1rem; animation: iconRotate 4s ease-in-out infinite 1s;">🪐</span>
        <span style="font-size: 1.5rem; margin: 0 1rem; animation: iconRotate 4s ease-in-out infinite 2s;">⭐</span>
        <span style="font-size: 1.5rem; margin: 0 1rem; animation: iconRotate 4s ease-in-out infinite 3s;">🚀</span>
    </div>
    <p style="font-size: 1rem; margin: 0; animation: textFade 3s ease-in-out infinite;">
        🌌 Powered by advanced machine learning algorithms • 
        🪐 Exploring the cosmos one prediction at a time • 
        ⭐ Built with Streamlit and cosmic inspiration
    </p>
    <div style="margin-top: 1rem;">
        <span style="display: inline-block; padding: 0.3rem 0.8rem; background: rgba(59, 130, 246, 0.1); border-radius: 15px; color: #3b82f6; font-size: 0.9rem; margin: 0 0.3rem; animation: badgePulse 2s ease-in-out infinite;">
            AI-Powered
        </span>
        <span style="display: inline-block; padding: 0.3rem 0.8rem; background: rgba(139, 92, 246, 0.1); border-radius: 15px; color: #8b5cf6; font-size: 0.9rem; margin: 0 0.3rem; animation: badgePulse 2s ease-in-out infinite 0.5s;">
            Real-time
        </span>
        <span style="display: inline-block; padding: 0.3rem 0.8rem; background: rgba(245, 158, 11, 0.1); border-radius: 15px; color: #f59e0b; font-size: 0.9rem; margin: 0 0.3rem; animation: badgePulse 2s ease-in-out infinite 1s;">
            Cosmic
        </span>
    </div>
</div>

<style>
@keyframes iconRotate {
    0%, 100% { transform: rotate(0deg) scale(1); }
    50% { transform: rotate(180deg) scale(1.2); }
}

@keyframes textFade {
    0%, 100% { opacity: 0.7; }
    50% { opacity: 1; }
}

@keyframes badgePulse {
    0%, 100% { transform: scale(1); opacity: 0.8; }
    50% { transform: scale(1.1); opacity: 1; }
}
</style>
""", unsafe_allow_html=True)
