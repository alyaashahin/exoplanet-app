Exoplanet Classification Models
==================================================

Training Date: 2025-09-30 04:25:59
Training Samples: 3178
Test Samples: 1363
Features: 34

How to Load and Use Models:
--------------------------------------------------
import joblib
import pandas as pd

# Load a model
model = joblib.load('best_model.pkl')

# Load the scaler
# scaler = joblib.load('scaler.pkl') -> not needed as values are already scaled

# Prepare your data (must have same features as training)
X_new = pd.DataFrame([{
    "koi_period": "select between its max and min value from below",
    "koi_period_err1": "select between its max and min value from below",
    "koi_time0bk": "select between its max and min value from below",
    "koi_time0bk_err1": "select between its max and min value from below",
    "koi_impact": "select between its max and min value from below",
    "koi_impact_err1": "select between its max and min value from below",
    "koi_impact_err2": "select between its max and min value from below",
    "koi_duration": "select between its max and min value from below",
    "koi_duration_err1": "select between its max and min value from below",
    "koi_depth": "select between its max and min value from below",
    "koi_depth_err1": "select between its max and min value from below",
    "koi_prad": "select between its max and min value from below",
    "koi_prad_err1": "select between its max and min value from below",
    "koi_prad_err2": "select between its max and min value from below",
    "koi_teq": "select between its max and min value from below",
    "koi_insol": "select between its max and min value from below",
    "koi_insol_err1": "select between its max and min value from below",
    "koi_model_snr": "select between its max and min value from below",
    "koi_steff": "select between its max and min value from below",
    "koi_steff_err1": "select between its max and min value from below",
    "koi_steff_err2": -"select between its max and min value from below",
    "koi_slogg": "select between its max and min value from below",
    "koi_slogg_err1": "select between its max and min value from below",
    "koi_slogg_err2": "select between its max and min value from below",
    "koi_srad_err1": "select between its max and min value from below",
    "koi_srad_err2": "select between its max and min value from below",
    "ra": "select between its max and min value from below",
    "dec": "select between its max and min value from below",
    "koi_kepmag": "select between its max and min value from below",
    "depth_to_srad": "select between its max and min value from below",
    "prad_to_srad_ratio": "select between its max and min value from below",
    "period_to_impact": "select between its max and min value from below",
    "log_insol": "select between its max and min value from below",
    "log_snr": "select between its max and min value from below"
}])


# Make predictions
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)

Features selected with max and min
--------------------------------------------------
                         min           max
koi_period         -0.389486  3.884597e+00
koi_period_err1    -0.249980  3.912221e+00
koi_time0bk        -0.509194  3.841147e+00
koi_time0bk_err1   -0.651946  3.741984e+00
koi_impact         -0.630537  3.631575e+00
koi_impact_err1    -0.666667  3.568627e+00
koi_impact_err2    -3.567234  7.243362e-01
koi_duration       -1.031911  3.611988e+00
koi_duration_err1  -0.692735  3.733467e+00
koi_depth          -0.572557  3.788191e+00
koi_depth_err1     -0.723214  3.843750e+00
koi_prad           -1.142857  3.691814e+00
koi_prad_err1      -0.723404  3.553191e+00
koi_prad_err2      -3.689655  7.241379e-01
koi_teq            -1.433444  3.570820e+00
koi_insol          -0.312844  3.735998e+00
koi_insol_err1     -0.261479  3.751848e+00
koi_model_snr      -0.675000  3.850000e+00
koi_steff          -3.652231  3.522310e+00
koi_steff_err1     -1.500000  3.550000e+00
koi_steff_err2     -3.494382  1.483146e+00
koi_slogg          -3.687500  3.363971e+00
koi_slogg_err1     -0.693333  3.658667e+00
koi_slogg_err2     -3.685185  1.037037e+00
koi_srad_err1      -0.941748  3.543689e+00
koi_srad_err2      -3.696721  8.032787e-01
ra                 -1.479197  1.324565e+00
dec                -1.327027  1.416869e+00
koi_kepmag         -3.649838  1.684677e+00
depth_to_srad      -0.479075  3.032855e+01
prad_to_srad_ratio -1.007994  1.379850e+01
period_to_impact   -0.239422  8.018737e+09
log_insol          -1.665119  9.453637e-01
log_snr            -1.799029  1.569439e+00


Model Performance Summary:
--------------------------------------------------
               Model  Accuracy  Precision   Recall  F1 Score  ROC-AUC
            CatBoost  0.833456   0.818811 0.860465  0.839121 0.906404 ===> Best model
            LightGBM  0.836390   0.819807 0.866279  0.842403 0.905017
       Random Forest  0.818048   0.814286 0.828488  0.821326 0.898789
   Gradient Boosting  0.823184   0.809986 0.848837  0.828957 0.897937
    Neural Net (MLP)  0.812179   0.800000 0.837209  0.818182 0.896494
                 LDA  0.804842   0.793872 0.828488  0.810811 0.878030
                 QDA  0.718269   0.666302 0.885174  0.760300 0.844830
 k-Nearest Neighbors  0.770360   0.740693 0.838663  0.786639 0.840171
  Naive Bayes (Bern)  0.738811   0.738506 0.747093  0.742775 0.799632
 Naive Bayes (Gauss)  0.495965   0.666667 0.002907  0.005789 0.783730
       Decision Tree  0.737344   0.736390 0.747093  0.741703 0.737250
      SGD Classifier  0.510638   0.515933 0.494186  0.504826 0.510797
 Logistic Regression  0.489362   0.494318 0.505814  0.500000 0.494447