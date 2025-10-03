# streamlit_app.py (ضعيه في جذر الريبو/الـ Space)
import os
import streamlit as st
import pandas as pd
import joblib

st.title("🚀 Exoplanet Classification App (Debug friendly)")

# عرض الملفات في مجلد التطبيق عشان تتأكدي الملف موجود
st.write("Files in app folder:", sorted(os.listdir(".")))

# محاولات تحميل الموديل من عدة أسماء شائعة
model = None
model_source = None
candidate_names = ["best_model.pkl", "catboost.pkl", "model.pkl", "final_model.pkl"]

for fname in candidate_names:
    if os.path.exists(fname):
        try:
            loaded = joblib.load(fname)
            # في حال حفظتي (model, feature_names) داخل ملف واحد
            if isinstance(loaded, tuple) and len(loaded) == 2:
                model, feature_names = loaded
            else:
                model = loaded
                feature_names = None
            model_source = fname
            st.success(f"Loaded model from: {fname}")
            break
        except Exception as e:
            st.error(f"Found {fname} but failed to load: {e}")

if model is None:
    st.error("No model file found in app root. Upload 'best_model.pkl' or 'catboost.pkl' via Files -> Upload files.")
    st.stop()

# لو لم تُحفظ أسماء الأعمدة مع الموديل نستخدم أسماء افتراضية من README
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

st.write(f"Using model file: **{model_source}**")

# نعرض بعض الحقول الأساسية للمستخدم ليدخلها (بقيّة الحقول تُملأ بصفر تلقائياً)
inputs = {}
inputs["koi_period"] = st.number_input("koi_period", value=1.0, format="%.6f")
inputs["koi_duration"] = st.number_input("koi_duration", value=1.0, format="%.6f")
inputs["koi_depth"] = st.number_input("koi_depth", value=1.0, format="%.6f")
inputs["koi_prad"] = st.number_input("koi_prad", value=1.0, format="%.6f")
inputs["koi_teq"] = st.number_input("koi_teq", value=500.0, format="%.6f")
inputs["koi_insol"] = st.number_input("koi_insol", value=1.0, format="%.6f")

X_new = pd.DataFrame([inputs])
X_new = X_new.reindex(columns=feature_names, fill_value=0)

if st.button("🔮 Predict"):
    try:
        pred = model.predict(X_new)
        proba = model.predict_proba(X_new) if hasattr(model, "predict_proba") else None

        prediction = pred[0] if hasattr(pred, "__len__") else pred
        st.write("### ✅ النتيجة:")
        if prediction == 1 or str(prediction).upper() == "CONFIRMED":
            st.success("هذا الكوكب **مرشح** (Candidate / CONFIRMED)")
        else:
            st.error("هذا الكوكب **ليس مرشح**")

        if proba is not None:
            st.write("### 🔢 احتمالات النموذج:")
            st.json(proba[0].tolist() if hasattr(proba[0], "tolist") else proba[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        # نطبع معلومات للتصحيح
        st.write("Input shape:", X_new.shape)
        st.write("Columns (input):", list(X_new.columns))
