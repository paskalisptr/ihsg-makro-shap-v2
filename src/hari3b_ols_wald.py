import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv("data_processed/dataset_final.csv", index_col=0, parse_dates=True).loc["2019-09-01":]

DOMESTIK = ["usdidr_lag1", "indonia_lag1", "bi_rate_lag1", "inflasi_lag1", "pdb_lag1"]
GLOBAL = ["sp500_lag1", "oil_wti_lag1", "dxy_lag1", "vix_lag1",
          "ust10y_lag1", "fed_funds_rate_lag1", "epu_lag1"]
TEKNIKAL = ["return_ihsg_lag1", "return_ihsg_lag2", "return_ihsg_lag3",
            "vol_roll5_lag1", "vol_roll20_lag1"]
FITUR = DOMESTIK + GLOBAL + TEKNIKAL
assert set(FITUR) == set(df.columns) - {"return_ihsg"}, "Ada fitur di dataset yang belum dikelompokkan!"

# Standardisasi z-score utk estimasi OLS (konsisten dgn metodologi v1)
X = (df[FITUR] - df[FITUR].mean()) / df[FITUR].std()
X = sm.add_constant(X)
y = df["return_ihsg"]

model = sm.OLS(y, X).fit()
print(model.summary())
print(f"\nR-squared: {model.rsquared:.4f}")

# ============================================================
# 4 dari 5 uji asumsi OLS (ADF sudah dikerjakan terpisah di hari3_eda.py)
# ============================================================
print("\n=== UJI ASUMSI OLS ===")

jb_stat, jb_p, _, _ = jarque_bera(model.resid)
print(f"Jarque-Bera (normalitas residual): stat={jb_stat:.2f}, p={jb_p:.4f}",
      "-> residual TIDAK normal (wajar utk return finansial, n besar)" if jb_p < 0.05 else "-> normal")

bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)
print(f"Breusch-Pagan (heteroskedastisitas): stat={bp_stat:.2f}, p={bp_p:.4f}")
pakai_hc3 = bp_p < 0.05
if pakai_hc3:
    print("-> Heteroskedastisitas terdeteksi, switch ke standard error robust HC3")
    model = sm.OLS(y, X).fit(cov_type="HC3")

dw = durbin_watson(model.resid)
print(f"Durbin-Watson (autokorelasi): {dw:.4f} (ideal ~2; <1.5 atau >2.5 indikasi masalah)")

vif_df = pd.DataFrame({
    "fitur": X.columns[1:],  # skip const
    "VIF": [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])]
})
print("\nVariance Inflation Factor (multikolinearitas, VIF>10 = perlu perhatian):")
print(vif_df.to_string(index=False))

# ============================================================
# RQ1: domestik jointly = 0 | RQ2: global jointly = 0
# ============================================================
def wald_group(nama_grup, kolom_grup):
    hipotesis = " = ".join(kolom_grup) + " = 0"
    hasil = model.wald_test(hipotesis, scalar=True)
    print(f"\nRQ (Wald) -- {nama_grup} jointly = 0: "
          f"F={hasil.statistic:.3f}, p={hasil.pvalue:.4f}",
          "-> SIGNIFIKAN" if hasil.pvalue < 0.05 else "-> tidak signifikan")
    return hasil.pvalue

p_domestik = wald_group("DOMESTIK (5 fitur)", DOMESTIK)
p_global = wald_group("GLOBAL (7 fitur)", GLOBAL)

# ============================================================
# RQ3: beda rata-rata |koefisien| domestik vs global -- kontras linear
# H0: mean(koef domestik) = mean(koef global)
# ============================================================
kontras = " + ".join([f"{1/len(DOMESTIK)}*{c}" for c in DOMESTIK]) + " - (" + \
          " + ".join([f"{1/len(GLOBAL)}*{c}" for c in GLOBAL]) + ")"
hasil_rq3 = model.t_test(kontras)
print(f"\nRQ3 -- beda rata-rata koefisien domestik vs global: "
      f"t={hasil_rq3.tvalue[0][0]:.3f}, p={hasil_rq3.pvalue:.4f}",
      "-> SIGNIFIKAN beda" if hasil_rq3.pvalue < 0.05 else "-> tidak signifikan beda")

print(f"\n=== RINGKASAN RQ1-3 (v2.0, 17 fitur) ===")
print(f"RQ1 (domestik signifikan?): p={p_domestik:.4f}")
print(f"RQ2 (global signifikan?):   p={p_global:.4f}")
print(f"RQ3 (beda domestik-global?): p={hasil_rq3.pvalue:.4f}")

# ============================================================
# VISUALISASI: koefisien OLS + interval kepercayaan, diwarnai per kelompok
# ============================================================
import matplotlib.pyplot as plt
from pathlib import Path
Path("outputs/figures").mkdir(parents=True, exist_ok=True)

warna_grup = {**{f: "#d62728" for f in DOMESTIK}, **{f: "#1f77b4" for f in GLOBAL},
              **{f: "#7f7f7f" for f in TEKNIKAL}}
coef = model.params.drop("const")
ci = model.conf_int().drop("const")
urutan = coef.reindex(coef.abs().sort_values().index).index  # urut dari efek terkecil

fig, ax = plt.subplots(figsize=(7, 7))
for i, f in enumerate(urutan):
    ax.errorbar(coef[f], i, xerr=[[coef[f] - ci.loc[f, 0]], [ci.loc[f, 1] - coef[f]]],
                fmt="o", color="black", ecolor=warna_grup[f], capsize=3)
ax.axvline(0, color="grey", lw=0.8, ls="--")
ax.set_yticks(range(len(urutan)))
ax.set_yticklabels(urutan)
ax.set_xlabel("Koefisien OLS (95% CI)")
ax.set_title("Koefisien OLS per fitur -- merah=domestik, biru=global, abu=teknikal")
plt.tight_layout()
plt.savefig("outputs/figures/ols_koefisien.png", dpi=150)
plt.close()
print("\nFigure tersimpan: outputs/figures/ols_koefisien.png")