import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

Path("outputs/figures").mkdir(parents=True, exist_ok=True)
Path("outputs/tables").mkdir(parents=True, exist_ok=True)
Path("outputs/models").mkdir(parents=True, exist_ok=True)

TRAIN_END = "2024-12-31"  # konsisten dgn v1 & hari3_eda.py

df = pd.read_csv("data_processed/dataset_final.csv", index_col=0, parse_dates=True).loc["2019-09-01":]
y, X = df["return_ihsg"], df.drop(columns="return_ihsg")
train, test = df.loc[:TRAIN_END], df.loc[TRAIN_END:].iloc[1:]
X_train, y_train = train[X.columns], train["return_ihsg"]
X_test, y_test = test[X.columns], test["return_ihsg"]
X_test.to_csv("outputs/tables/X_test.csv")
y_test.to_csv("outputs/tables/y_test.csv")
print(f"Train: {len(X_train)} ({X_train.index.min()} s/d {X_train.index.max()})")
print(f"Test:  {len(X_test)} ({X_test.index.min()} s/d {X_test.index.max()})")

tscv = TimeSeriesSplit(n_splits=5)

# ============================================================
# Naif (baseline) + Linear Regression + RF (tuned) + XGBoost (tuned)
# ============================================================
pred = {}
pred["naif"] = np.zeros(len(y_test))  # selalu tebak 0 -> arah "naik" (sign(0)==0, dihitung salah semua)

lr = LinearRegression().fit(X_train, y_train)
pred["lr"] = lr.predict(X_test)

rf_grid = {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 8, None],
           "min_samples_leaf": [1, 5, 10], "max_features": ["sqrt", 0.5, 1.0]}
rf_search = RandomizedSearchCV(RandomForestRegressor(random_state=42), rf_grid, n_iter=20,
                                cv=tscv, scoring="neg_root_mean_squared_error", random_state=42, n_jobs=-1)
rf_search.fit(X_train, y_train)
rf = rf_search.best_estimator_
pred["rf"] = rf.predict(X_test)
print(f"\nRF best params: {rf_search.best_params_}")

xgb_grid = {"n_estimators": [100, 200, 300], "max_depth": [2, 3, 4, 5],
            "learning_rate": [0.01, 0.03, 0.1], "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0]}
xgb_search = RandomizedSearchCV(XGBRegressor(random_state=42), xgb_grid, n_iter=20,
                                 cv=tscv, scoring="neg_root_mean_squared_error", random_state=42, n_jobs=-1)
xgb_search.fit(X_train, y_train)
xgb = xgb_search.best_estimator_
pred["xgb"] = xgb.predict(X_test)
print(f"XGB best params: {xgb_search.best_params_}")

joblib.dump(lr, "outputs/models/lr.joblib")
joblib.dump(rf, "outputs/models/rf.joblib")
joblib.dump(xgb, "outputs/models/xgb.joblib")

# ============================================================
# Evaluasi: RMSE, MAE, directional accuracy
# ============================================================
hasil = []
for nama, p in pred.items():
    hasil.append({
        "model": nama,
        "rmse": np.sqrt(mean_squared_error(y_test, p)),
        "mae": mean_absolute_error(y_test, p),
        "directional_accuracy": (np.sign(p) == np.sign(y_test)).mean(),
    })
hasil_df = pd.DataFrame(hasil)
hasil_df.to_csv("outputs/tables/evaluasi_model.csv", index=False)
print("\n=== EVALUASI ===")
print(hasil_df.to_string(index=False))

# ============================================================
# VISUALISASI 1: perbandingan directional accuracy antar model
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(hasil_df["model"], hasil_df["directional_accuracy"], color="#1f77b4")
ax.axhline(0.5, color="red", ls="--", lw=1, label="Tebak acak (50%)")
ax.set_ylabel("Directional Accuracy")
ax.set_title("Perbandingan Akurasi Arah Antar Model (v2.0)")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/figures/perbandingan_akurasi_model.png", dpi=150)
plt.close()

# ============================================================
# VISUALISASI 2: actual vs predicted (model terbaik berdasar directional accuracy)
# ============================================================
model_terbaik = hasil_df.loc[hasil_df["directional_accuracy"].idxmax(), "model"]
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(y_test.index, y_test, label="Aktual", lw=0.8, color="black")
ax.plot(y_test.index, pred[model_terbaik], label=f"Prediksi ({model_terbaik})", lw=0.8, alpha=0.7)
ax.set_ylabel("Return IHSG")
ax.set_title(f"Aktual vs Prediksi -- Model Terbaik: {model_terbaik}")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/figures/aktual_vs_prediksi.png", dpi=150)
plt.close()

# ============================================================
# VISUALISASI 3: feature importance bawaan RF & XGBoost (BUKAN SHAP --
# SHAP menyusul di Fase 9, ini cuma cek cepat/preview)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, (nama, mdl) in zip(axes, [("Random Forest", rf), ("XGBoost", xgb)]):
    imp = pd.Series(mdl.feature_importances_, index=X.columns).sort_values()
    ax.barh(imp.index, imp.values)
    ax.set_title(f"Feature Importance -- {nama}")
plt.tight_layout()
plt.savefig("outputs/figures/feature_importance_preview.png", dpi=150)
plt.close()

print(f"\nModel terbaik: {model_terbaik}")
print("Semua figure tersimpan di outputs/figures/, model tersimpan di outputs/models/")