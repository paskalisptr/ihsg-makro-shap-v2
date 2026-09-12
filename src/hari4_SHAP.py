import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
from pathlib import Path
from scipy.stats import mannwhitneyu, binomtest
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error

Path("outputs/figures").mkdir(parents=True, exist_ok=True)
Path("outputs/tables").mkdir(parents=True, exist_ok=True)

DOMESTIK = ["usdidr_lag1", "indonia_lag1", "bi_rate_lag1", "inflasi_lag1", "pdb_lag1"]
GLOBAL = ["sp500_lag1", "oil_wti_lag1", "dxy_lag1", "vix_lag1",
          "ust10y_lag1", "fed_funds_rate_lag1", "epu_lag1"]
TEKNIKAL = ["return_ihsg_lag1", "return_ihsg_lag2", "return_ihsg_lag3",
            "vol_roll5_lag1", "vol_roll20_lag1"]

# Anotasi visual SAJA -- bukan dasar uji statistik (sama seperti v1, sekarang +COVID)
CRISIS_WINDOWS = [
    ("2020-02-01", "2020-04-30"),
    ("2025-03-01", "2025-04-30"),
    ("2026-01-28", "2026-05-31"),
]

# ============================================================
# Muat data & model dari Fase 7
# ============================================================
df = pd.read_csv("data_processed/dataset_final.csv", index_col=0, parse_dates=True).loc["2019-09-01":]
y, X = df["return_ihsg"], df.drop(columns="return_ihsg")
train = df.loc[:"2024-12-31"]
X_train, y_train = train[X.columns], train["return_ihsg"]

X_test = pd.read_csv("outputs/tables/X_test.csv", index_col=0, parse_dates=True)
y_test = pd.read_csv("outputs/tables/y_test.csv", index_col=0, parse_dates=True).iloc[:, 0]

fitted = {name: joblib.load(f"outputs/models/{name}.joblib") for name in ["lr", "rf", "xgb"]}

# ============================================================
# SHAP per model
# ============================================================
masker_lr = shap.maskers.Independent(X_train, max_samples=len(X_train))
explainers = {
    "lr": shap.LinearExplainer(fitted["lr"], masker_lr),
    "rf": shap.TreeExplainer(fitted["rf"]),
    "xgb": shap.TreeExplainer(fitted["xgb"]),
}

shap_values = {}
for name, expl in explainers.items():
    sv = expl.shap_values(X_test)
    shap_values[name] = pd.DataFrame(sv, columns=X_test.columns, index=X_test.index)
    mean_abs = shap_values[name].abs().mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(mean_abs.index[::-1], mean_abs.values[::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Kepentingan fitur -- {name.upper()} (v2.0, 17 fitur)")
    plt.tight_layout()
    plt.savefig(f"outputs/figures/shap_importance_{name}.png", dpi=150)
    plt.close()

    print(f"\nSHAP {name} -- 5 fitur teratas:\n{mean_abs.head()}")

    fig, ax = plt.subplots(figsize=(7, 6))
    shap.summary_plot(shap_values[name].values, X_test, show=False, plot_size=None)
    plt.tight_layout()
    plt.savefig(f"outputs/figures/shap_summary_{name}.png", dpi=150)
    plt.close()

# ============================================================
# Rolling-window SHAP: dominansi global vs domestik dari waktu ke waktu
# (teknikal dikeluarkan dari perbandingan ini -- fokus RQ4-5 murni domestik-global)
# Model utama: XGBoost, konsisten dgn v1
# ============================================================
sv_xgb = shap_values["xgb"]
domestik_abs = sv_xgb[DOMESTIK].abs().sum(axis=1)
global_abs = sv_xgb[GLOBAL].abs().sum(axis=1)

WINDOW = 20
roll_domestik = domestik_abs.rolling(WINDOW).mean()
roll_global = global_abs.rolling(WINDOW).mean()

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(roll_domestik.index, roll_domestik, label="Domestik (rolling 20 hari)", lw=1)
ax.plot(roll_global.index, roll_global, label="Global (rolling 20 hari)", lw=1)
for start, end in CRISIS_WINDOWS:
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.12, color="red")
ax.set_ylabel("Rata-rata |SHAP| bergulir")
ax.set_title("Dominansi SHAP domestik vs global -- v2.0 (merah = periode ilustratif, bukan dasar uji)")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/figures/shap_rolling_domestik_vs_global.png", dpi=150)
plt.close()

# ============================================================
# Mann-Whitney U -- uji utama RQ5, berbasis split volatilitas (objektif, EDA-driven)
# vol pakai shift(1) supaya TIDAK termasuk return hari ini sendiri
# ============================================================
vol20 = df["return_ihsg"].rolling(20).std().shift(1)
vol_test = vol20.reindex(y_test.index)
mask_high_vol = (vol_test > vol_test.median()).values

diff_shap = (global_abs - domestik_abs).values
u_stat, p_value = mannwhitneyu(diff_shap[mask_high_vol], diff_shap[~mask_high_vol], alternative="two-sided")
print(f"\nMann-Whitney U (dominansi global-domestik, vol tinggi vs rendah): U={u_stat:.1f}, p={p_value:.4f}")

n1, n2 = int(mask_high_vol.sum()), int((~mask_high_vol).sum())
r_rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

mwu_result = pd.DataFrame([{
    "pembanding": "vol_tinggi_vs_rendah", "U": u_stat, "p_value": p_value,
    "effect_size_r": r_rank_biserial,
    "median_diff_vol_tinggi": float(np.median(diff_shap[mask_high_vol])),
    "median_diff_vol_rendah": float(np.median(diff_shap[~mask_high_vol])),
    "n_tinggi": n1, "n_rendah": n2,
}])
mwu_result.to_csv("outputs/tables/mann_whitney_shap_v2.csv", index=False)
print(mwu_result.T)

# ============================================================
# Ablation study: domestik(+teknikal) vs global(+teknikal) vs penuh
# Teknikal disertakan di semua skenario sbg kontrol -- lihat catatan desain.
# ============================================================
feature_sets = {
    "domestik_saja": DOMESTIK + TEKNIKAL,
    "global_saja": GLOBAL + TEKNIKAL,
    "penuh": list(X.columns),
}
ablation_rows = []
for fs_name, cols in feature_sets.items():
    for model_name in ["rf", "xgb"]:
        model = clone(fitted[model_name])
        model.fit(X_train[cols], y_train)
        pred = model.predict(X_test[cols])
        ablation_rows.append({
            "fitur": fs_name, "model": model_name,
            "rmse": np.sqrt(mean_squared_error(y_test, pred)),
            "mae": mean_absolute_error(y_test, pred),
            "directional_accuracy": (np.sign(pred) == np.sign(y_test)).mean(),
        })

ablation_df = pd.DataFrame(ablation_rows)
print("\nHasil ablation (v2.0):\n", ablation_df)
ablation_df.to_csv("outputs/tables/ablation_study_v2.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4))
piv = ablation_df.pivot(index="fitur", columns="model", values="directional_accuracy")
piv = piv.reindex(["domestik_saja", "global_saja", "penuh"])
piv.plot(kind="bar", ax=ax)
ax.axhline(0.5, color="red", ls="--", lw=1, label="Tebak acak")
ax.set_ylabel("Directional Accuracy")
ax.set_title("Ablation Study v2.0 -- Domestik+Teknikal vs Global+Teknikal vs Penuh")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/figures/ablation_v2.png", dpi=150)
plt.close()

# ============================================================
# Uji Binomial: directional accuracy signifikan beda dari 50%?
# ============================================================
pred_xgb_full = fitted["xgb"].predict(X_test)
pred_rf_full = fitted["rf"].predict(X_test)
pred_lr_full = fitted["lr"].predict(X_test)

binom_cases = [
    ("XGBoost keseluruhan", np.sign(pred_xgb_full) == np.sign(y_test.values)),
    ("XGBoost vol tinggi", (np.sign(pred_xgb_full) == np.sign(y_test.values))[mask_high_vol]),
    ("XGBoost vol rendah", (np.sign(pred_xgb_full) == np.sign(y_test.values))[~mask_high_vol]),
    ("Random Forest keseluruhan", np.sign(pred_rf_full) == np.sign(y_test.values)),
    ("Linear Regression keseluruhan", np.sign(pred_lr_full) == np.sign(y_test.values)),
]

binom_rows = []
print("\n=== Uji Binomial (v2.0): akurasi arah vs tebak acak (50%) ===")
for name, correct in binom_cases:
    k, n = int(correct.sum()), len(correct)
    res = binomtest(k, n, 0.5, alternative="greater")
    status = "SIGNIFIKAN" if res.pvalue < 0.05 else "tidak signifikan"
    print(f"{name:32s} k={k:3d}/{n:3d}  akurasi={k/n:.4f}  p={res.pvalue:.4f}  {status}")
    binom_rows.append({"segmen": name, "k": k, "n": n, "akurasi": k / n, "p_value": res.pvalue, "status": status})

binom_df = pd.DataFrame(binom_rows)
binom_df.to_csv("outputs/tables/uji_binomial_v2.csv", index=False)
print("\nSemua figure/tabel tersimpan di outputs/figures/ dan outputs/tables/")