import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.stattools import adfuller

Path("outputs/figures").mkdir(parents=True, exist_ok=True)
Path("outputs/tables").mkdir(parents=True, exist_ok=True)

# Dua episode volatilitas 2025-2026 yang relevan untuk Pembahasan.
# Tanggal Mar-Apr 2025 diverifikasi via web search: trading halt IHSG
# terjadi 18 Maret 2025 dan 8 & 11 April 2025 (tarif resiprokal Trump,
# diumumkan 2 April 2025). Persempit rentang ini kalau perlu presisi lebih.
CRISIS_WINDOWS = [
    ("2020-02-01", "2020-04-30"),  # COVID-19: trading halt berulang Maret 2020 (BEI, 6x dlm sebulan)
    ("2025-03-01", "2025-04-30"),  # tarif Trump "Liberation Day" + Danantara
    ("2026-01-28", "2026-05-31"),  # MSCI reclass / BI rate darurat
]
TRAIN_END = "2024-12-31"

df = pd.read_csv("data_processed/dataset_final.csv", index_col=0, parse_dates=True).loc["2019-09-01":]
print("Dataset final:", df.shape, df.index.min(), "s/d", df.index.max())

y = df["return_ihsg"]
X = df.drop(columns="return_ihsg")
train, test = df.loc[:TRAIN_END], df.loc[TRAIN_END:].iloc[1:]

print("\n=== Statistik deskriptif ===")
print(df.describe().T)

# ============================================================
# Uji Augmented Dickey-Fuller (stasioneritas) -- sekali saja.
# H0: ada unit root (non-stasioner). p<0.05 -> tolak H0 -> stasioner.
# WAJIB sebelum OLS: regresi antar variabel non-stasioner berisiko
# spurious regression (Granger & Newbold, 1974 -- verifikasi sitasi
# persis sebelum dikutip di naskah).
# vix_lag1 dipertahankan level (bukan diff) -- satu-satunya fitur yang
# stasioneritasnya perlu dibuktikan empiris, bukan diasumsikan dari teori.
# ============================================================
adf_df = pd.DataFrame(
    {"variabel": c, "adf_stat": (r := adfuller(df[c].dropna()))[0], "p_value": r[1],
     "status": "Stasioner" if r[1] < 0.05 else "TIDAK STASIONER"}
    for c in df.columns
)
print("\n=== Uji ADF (stasioneritas) ===")
print(adf_df.to_string(index=False))
adf_df.to_csv("outputs/tables/uji_adf.csv", index=False)
if (adf_df["status"] == "TIDAK STASIONER").any():
    print("\nPERINGATAN: ada variabel tidak stasioner -- OLS/RQ1-3 berisiko "
          "spurious regression, perlu differencing tambahan sebelum lanjut.")


def mark_crisis(ax):
    for start, end in CRISIS_WINDOWS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.2, color="red")


# ============================================================
# Outlier ekstrem (>3xIQR) -- dihitung sekali, dipakai untuk tabel & semua plot
# ============================================================
q1, q3 = y.quantile([0.25, 0.75])
iqr = q3 - q1
is_outlier = (y < q1 - 3 * iqr) | (y > q3 + 3 * iqr)
outliers = df[is_outlier]
print(f"\nOutlier ekstrem (>3xIQR): {len(outliers)} hari, "
      f"{len(outliers.loc[:TRAIN_END])} di train, {len(outliers.loc['2025-01-01':])} di test")
print(outliers[["return_ihsg"]].sort_values("return_ihsg"))
outliers[["return_ihsg"]].to_csv("outputs/tables/outlier_return.csv")

# ============================================================
# Return & volatilitas bergulir, krisis ditandai, outlier ditandai (BARU)
# ============================================================
vol20 = y.rolling(20).std()
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
axes[0].plot(df.index, y, lw=0.6)
axes[0].scatter(df.index[is_outlier], y[is_outlier], color="red", zorder=5, label="Outlier >3xIQR")
axes[0].legend()
mark_crisis(axes[0])
axes[0].set_ylabel("Return harian")
axes[1].plot(vol20.index, vol20, lw=0.9, color="darkorange")
mark_crisis(axes[1])
axes[1].set_ylabel("Volatilitas bergulir 20 hari")
plt.tight_layout()
plt.savefig("outputs/figures/eda_return_dan_volatilitas.png", dpi=150)
plt.close()

# ============================================================
# Histogram distribusi return
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(y, bins=60)
ax.set_xlabel("Return harian IHSG")
plt.tight_layout()
plt.savefig("outputs/figures/eda_return_histogram.png", dpi=150)
plt.close()

# ============================================================
# Heatmap korelasi antar fitur (cek multikolinearitas kasar sebelum OLS)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
corr = X.corr()
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticks(range(len(corr.columns)))
ax.set_yticklabels(corr.columns)
plt.colorbar(im)
plt.tight_layout()
plt.savefig("outputs/figures/eda_correlation_heatmap.png", dpi=150)
plt.close()

import math

def grid_kosong(n, ncols=5, lebar_per_kolom=3.2, tinggi_per_baris=3):
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * lebar_per_kolom, nrows * tinggi_per_baris))
    return fig, list(axes.flat)  # materialize sekali jadi list -- aman dipakai berkali-kali,
                                   # beda dengan axes.flat mentah yang iterator sekali-pakai

# ============================================================
# BARU -- Boxplot tiap fitur (deteksi outlier per-fitur, bukan cuma target)
# ============================================================
fig, axes_flat = grid_kosong(len(X.columns))
for ax, col in zip(axes_flat, X.columns):
    ax.boxplot(X[col].dropna())
    ax.set_title(col, fontsize=9)
for ax in list(axes_flat)[len(X.columns):]:
    ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/figures/eda_boxplot_per_fitur.png", dpi=150)
plt.close()

# ============================================================
# BARU -- Time series tiap fitur (cek lompatan/pola aneh visual per fitur)
# ============================================================
fig, axes_flat = grid_kosong(len(X.columns), ncols=3, tinggi_per_baris=2.2)
for ax, col in zip(axes_flat, X.columns):
    ax.plot(X.index, X[col], lw=0.5)
    mark_crisis(ax)
    ax.set_title(col, fontsize=9)
for ax in list(axes_flat)[len(X.columns):]:
    ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/figures/eda_timeseries_per_fitur.png", dpi=150)
plt.close()

# ============================================================
# BARU -- Distribusi train vs test (cek apakah test "beda dunia" dari train,
# penting krn desain walk-forward -- kalau beda jauh, wajar akurasi test turun)
# ============================================================
kolom_cek = ["return_ihsg"] + list(X.columns)
fig, axes_flat = grid_kosong(len(kolom_cek))
for ax, col in zip(axes_flat, kolom_cek):
    ax.hist(train[col].dropna(), bins=30, alpha=0.5, density=True, label="Train")
    ax.hist(test[col].dropna(), bins=30, alpha=0.5, density=True, label="Test")
    ax.set_title(col, fontsize=9)
for ax in list(axes_flat)[len(kolom_cek):]:
    ax.axis("off")
list(axes_flat)[0].legend(fontsize=7)
plt.tight_layout()
plt.savefig("outputs/figures/eda_train_vs_test.png", dpi=150)
plt.close()

print("\nSemua figure tersimpan di outputs/figures/")