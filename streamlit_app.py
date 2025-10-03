import streamlit as st
import pandas as pd
import joblib

# تحميل الموديل
model = joblib.load("best_model.pkl")

# أسماء الأعمدة زي التدريب (34 feature)
feature_names = [
    "koi_period", "koi_period_err1", "koi_time0bk", "koi_time0bk_err1",
    "koi_impact", "koi_impact_err1", "koi_impact_err2", "koi_duration",
    "koi_duration_err1", "koi_depth", "koi_depth_err1", "koi_prad",
    "koi_prad_err1", "koi_prad_err2", "koi_teq", "koi_insol",
    "koi_insol_err1", "koi_model_snr", "koi_steff", "koi_steff_err1",
    "koi_steff_err2", "koi_slogg", "koi_slogg_err1", "koi_slogg_err2",
    "koi_srad_err1", "koi_srad_err2", "ra", "dec", "koi_kepmag",
    "depth_to_srad", "prad_to_srad_ratio", "period_to_impact",
    "log_insol", "log_snr"
]

st.title("🚀 Exoplanet Classification App")
st.write("أدخلي القيم المطلوبة (بعضها فقط، والباقي هنسيبه 0):")

# نخلي المستخدم يدخل أهم القيم
inputs = {}
inputs["koi_period"] = st.number_input("koi_period", value=1.0)
inputs["koi_duration"] = st.number_input("koi_duration", value=1.0)
inputs["koi_depth"] = st.number_input("koi_depth", value=1.0)
inputs["koi_prad"] = st.number_input("koi_prad", value=1.0)
inputs["koi_teq"] = st.number_input("koi_teq", value=500.0)
inputs["koi_insol"] = st.number_input("koi_insol", value=1.0)

# نجهز DataFrame
X_new = pd.DataFrame([inputs])

# نرتب الأعمدة ونملى الباقي بـ 0
X_new = X_new.reindex(columns=feature_names, fill_value=0)

# زر التنبؤ
if st.button("🔮 Predict"):
    prediction = model.predict(X_new)[0]
    probabilities = model.predict_proba(X_new)[0]

    st.write("### ✅ النتيجة:")
    if prediction == 1:
        st.success("هذا الكوكب **مرشح** (Candidate)")
    else:
        st.error("هذا الكوكب **ليس مرشح**")

    st.write("### 🔢 احتمالات النموذج:")
    st.json({
        "Not Candidate": f"{probabilities[0]:.2f}",
        "Candidate": f"{probabilities[1]:.2f}"
    })
