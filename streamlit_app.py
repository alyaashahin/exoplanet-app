import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Exoplanet Predictor", layout="wide")

st.title("🔭 Exoplanet Predictor")
st.markdown("Provide feature values (or randomize them) to predict with your CatBoost model.")

EXPLANATIONS = {
    # Main KOI features
    "Orbital Period (days)": "How many days the planet takes to go around its star. Like Earth takes 365 days to go around the Sun!",
    "Orbital Period Error (+)": "How much we might be off when guessing the orbit time.",
    "Time of Transit (BJD)": "The exact moment when the planet walks in front of its star—like crossing in front of a flashlight.",
    "Time of Transit Error (+)": "How much our 'crossing time' guess could be off.",
    "Transit Duration (hrs)": "How many hours the planet is blocking the star. Short peek 👀 or long peek!",
    "Transit Duration Error (+)": "How much our timing could be off for how long the planet blocks the star.",
    "Transit Depth (ppm)": "How much dimmer the star looks when the planet blocks it. Big dip = bigger planet.",
    "Transit Depth Error (+)": "How much our dimness guess could be off.",
    "Planet Radius (Earth radii)": "How big the planet is compared to Earth. 2 means it's twice as big as Earth 🌍.",
    "Planet Radius Error (+)": "How much we might be off when guessing the planet’s size (too big?).",
    "Planet Radius Error (-)": "How much we might be off when guessing the planet’s size (too small?).",
    "Equilibrium Temperature (K)": "How hot or cold the planet might be if it had no air blanket. 🔥🥶",
    "Insolation Flux (Earth flux)": "How much starlight the planet gets compared to Earth. More light = toastier planet.",
    "Insolation Flux Error (+)": "How much our 'sunlight guess' could be off.",
    "Transit Model SNR": "How clear the planet signal is compared to noise. High SNR = strong signal 💡.",
    "Stellar Effective Temp (K)": "How hot the star’s surface is. Hotter star = brighter and bluer.",
    "Stellar Effective Temp Error (+)": "How much we might be off in the star’s hotness (too hot?).",
    "Stellar Effective Temp Error (-)": "How much we might be off in the star’s hotness (too cold?).",
    "Surface Gravity (log g)": "How strong the star’s gravity is. Strong pull = you’d feel super heavy!",
    "Surface Gravity Error (+)": "How much we could be wrong in guessing how strong the star pulls (too strong?).",
    "Surface Gravity Error (-)": "How much we could be wrong in guessing how strong the star pulls (too weak?).",
    "Stellar Radius Error (+)": "How much bigger the star might really be than our guess.",
    "Stellar Radius Error (-)": "How much smaller the star might really be than our guess.",
    "RA (deg)": "Where the star sits left-right on the sky map. Like longitude for stars.",
    "Dec (deg)": "Where the star sits up-down on the sky map. Like latitude for stars.",
    "Kepler Magnitude": "How bright the star looks to the Kepler telescope 👀.",

    # Extra KOI parameters
    "Impact Parameter": "How much the planet’s path misses the middle of the star. 0 = right in front, 1 = just skimming the edge.",
    "Impact Parameter Error (+)": "How much our guess could be too high for that skim factor.",
    "Impact Parameter Error (-)": "How much our guess could be too low for that skim factor.",

    # Engineered features (your extra math tricks)
    "Depth / Stellar Radius": "How big the dip in light is compared to the star’s size. Bigger dip = bigger planet relative to star.",
    "Planet / Stellar Radius Ratio": "How big the planet is compared to its star. Like saying 'the planet is a marble, the star is a beach ball'.",
    "Period / Impact Parameter": "A mix of how long the orbit takes and how much the planet misses the center. More mathy combo ⚡.",
    "Log(1 + Insolation Flux)": "A math trick to shrink sunlight values into a friendlier scale.",
    "Log(1 + Transit SNR)": "A math trick to shrink signal-to-noise into a tidier number.",
}


# Mapping: Pretty names -> Model feature names
NAME_MAP = {
    "Orbital Period (days)": "koi_period",
    "Orbital Period Error (+)": "koi_period_err1",
    "Time of Transit (BJD)": "koi_time0bk",
    "Transit Duration (hrs)": "koi_duration",
    "Transit Depth (ppm)": "koi_depth",
    "Planet Radius (Earth radii)": "koi_prad",
    "Equilibrium Temperature (K)": "koi_teq",
    "Insolation Flux (Earth flux)": "koi_insol",
    "Stellar Effective Temp (K)": "koi_steff",
    "Surface Gravity (log g)": "koi_slogg",
    "Stellar Radius Error (+)": "koi_srad_err1",
    "RA (deg)": "ra",
    "Dec (deg)": "dec",
    "Kepler Magnitude": "koi_kepmag",

    # Extra KOI parameters
    "Time of Transit Error (+)": "koi_time0bk_err1",
    "Impact Parameter": "koi_impact",
    "Impact Parameter Error (+)": "koi_impact_err1",
    "Impact Parameter Error (-)": "koi_impact_err2",
    "Transit Duration Error (+)": "koi_duration_err1",
    "Transit Depth Error (+)": "koi_depth_err1",
    "Planet Radius Error (+)": "koi_prad_err1",
    "Planet Radius Error (-)": "koi_prad_err2",
    "Insolation Flux Error (+)": "koi_insol_err1",
    "Transit Model SNR": "koi_model_snr",
    "Stellar Effective Temp Error (+)": "koi_steff_err1",
    "Stellar Effective Temp Error (-)": "koi_steff_err2",
    "Surface Gravity Error (+)": "koi_slogg_err1",
    "Surface Gravity Error (-)": "koi_slogg_err2",
    "Stellar Radius Error (-)": "koi_srad_err2",

    # Engineered features:
    "Depth / Stellar Radius": "depth_to_srad",
    "Planet / Stellar Radius Ratio": "prad_to_srad_ratio",
    "Period / Impact Parameter": "period_to_impact",
    "Log(1 + Insolation Flux)": "log_insol",
    "Log(1 + Transit SNR)": "log_snr",
}

# Define feature ranges (UI side, pretty names)
FEATURE_RANGES = {
    "Orbital Period (days)": (-0.389486, 3.884597),
    "Orbital Period Error (+)": (-0.249980, 3.912221),
    "Time of Transit (BJD)": (-0.509194, 3.841147),
    "Time of Transit Error (+)": (-0.651946, 3.741984),
    "Impact Parameter": (-0.630537, 3.631575),
    "Impact Parameter Error (+)": (-0.666667, 3.568627),
    "Impact Parameter Error (-)": (-3.567234, 0.7243362),
    "Transit Duration (hrs)": (-1.031911, 3.611988),
    "Transit Duration Error (+)": (-0.692735, 3.733467),
    "Transit Depth (ppm)": (-0.572557, 3.788191),
    "Transit Depth Error (+)": (-0.723214, 3.843750),
    "Planet Radius (Earth radii)": (-1.142857, 3.691814),
    "Planet Radius Error (+)": (-0.723404, 3.553191),
    "Planet Radius Error (-)": (-3.689655, 0.7241379),
    "Equilibrium Temperature (K)": (-1.433444, 3.570820),
    "Insolation Flux (Earth flux)": (-0.312844, 3.735998),
    "Insolation Flux Error (+)": (-0.261479, 3.751848),
    "Transit Model SNR": (-0.675000, 3.850000),
    "Stellar Effective Temp (K)": (-3.652231, 3.522310),
    "Stellar Effective Temp Error (+)": (-1.500000, 3.550000),
    "Stellar Effective Temp Error (-)": (-3.494382, 1.483146),
    "Surface Gravity (log g)": (-3.687500, 3.363971),
    "Surface Gravity Error (+)": (-0.693333, 3.658667),
    "Surface Gravity Error (-)": (-3.685185, 1.037037),
    "Stellar Radius Error (+)": (-0.941748, 3.543689),
    "Stellar Radius Error (-)": (-3.696721, 0.8032787),
    "RA (deg)": (-1.479197, 1.324565),
    "Dec (deg)": (-1.327027, 1.416869),
    "Kepler Magnitude": (-3.649838, 1.684677),

    # Engineered features:
    "Depth / Stellar Radius": (-0.479075, 30.32855),
    "Planet / Stellar Radius Ratio": (-1.007994, 13.79850),
    "Period / Impact Parameter": (-0.239422, 8.018737e9),
    "Log(1 + Insolation Flux)": (-1.665119, 0.9453637),
    "Log(1 + Transit SNR)": (-1.799029, 1.569439),
}


# Initialize session state for feature values
if "feature_values" not in st.session_state:
    st.session_state.feature_values = {
        k: float((v[0] + v[1]) / 2.0) for k, v in FEATURE_RANGES.items()
    }

# Utility functions
def random_value_between(key):
    lo, hi = FEATURE_RANGES[key]
    return float(np.random.uniform(lo, hi))

def assign_random(feature):
    st.session_state.feature_values[feature] = random_value_between(feature)

def assign_random_all():
    for k in FEATURE_RANGES.keys():
        st.session_state.feature_values[k] = random_value_between(k)

# Callback function for individual randomize buttons
def randomize_callback(feature):
    st.session_state.feature_values[feature] = random_value_between(feature)

# Load CatBoost model directly
model = None
if os.path.exists("catboost.pkl"):
    model = joblib.load("catboost.pkl")
else:
    st.error("❌ Could not find `catboost.pkl`. Please place it in the same directory as this script.")

# Input grid
st.header("🛠️ Feature Inputs")

# Handle randomize button clicks first
for feat in FEATURE_RANGES.keys():
    if f"rand_{feat}" in st.session_state and st.session_state[f"rand_{feat}"]:
        st.session_state.feature_values[feat] = random_value_between(feat)
        st.session_state[f"rand_{feat}"] = False
        st.rerun()

# ✅ make 2 columns (so features split nicely)
cols = st.columns(2)

for i, (feat, (lo, hi)) in enumerate(FEATURE_RANGES.items()):
    c1, c2 = cols[i % 2].columns([4, 1])  # main box + dice button
    
    # Create the number input with the current value from session state
    val = c1.number_input(
        label=feat,
        min_value=float(lo),
        max_value=float(hi),
        value=float(st.session_state.feature_values.get(feat, (lo + hi) / 2.0)),
        format="%.6g",
        key=f"input_{feat}",
    )
    # Update session state with the current input value
    st.session_state.feature_values[feat] = float(val)
    
    # Randomize button
    if c2.button("🎲", key=f"rand_{feat}", help="Randomize this feature"):
        st.session_state[f"rand_{feat}"] = True
        st.rerun()

    # 👶 Childish explanation in an expander
    with c1.expander("Click for explanation"):
        st.write(EXPLANATIONS.get(feat, "No simple story for this one yet."))


st.markdown("---")
if st.button("🎲 Randomize All Features"):
    assign_random_all()
    st.rerun()  # ✅ correct call for modern Streamlit

# Prepare DataFrame for model
# Prepare DataFrame for model
X_dict = {}
for pretty_name, value in st.session_state.feature_values.items():
    raw_name = NAME_MAP.get(pretty_name, pretty_name)  # fall back to same if not found
    X_dict[raw_name] = [float(value)]

X_df = pd.DataFrame.from_dict(X_dict)

st.subheader("📊 Input Preview (Model Input)")
st.dataframe(X_df, use_container_width=True)

# Predictions
if model is not None and st.button("🚀 Predict"):
    try:
        # Align feature order with model expectations
        expected_features = model.feature_names_
        X_df = X_df.reindex(columns=expected_features)

        preds = model.predict(X_df)
        st.success(f"✅ Prediction: {preds[0]}")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_df)[0]
            prob_df = pd.DataFrame([probs], columns=[f"Class {i}" for i in range(len(probs))])
            st.subheader("📈 Prediction Probabilities")
            st.bar_chart(prob_df.T)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
