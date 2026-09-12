import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import binomtest

Path("outputs/figures").mkdir(parents=True, exist_ok=True)
Path("outputs/tables").mkdir(parents=True, exist_ok=True)
Path("outputs/models").mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")  # SARIMAX suka cerewet soal konvergensi minor, tidak fatal

df = pd.read_csv("data_processed/dataset_final.csv", index_col=0, parse_dates=True).loc["2019-09-01":]
y, X = df["return_ihsg"], df.drop(columns="return_ihsg")
train, test = df.loc[:"2024-12-31"], df.loc["2024-12-31":].iloc[1:]
y_train, X_train = train["return_ihsg"], train[X.columns]
y_test, X_test = test["return_ihsg"], test[X.columns]
tanggal_test = X_test.index  # simpan buat grafik nanti

# SARIMAX butuh index kontinu (RangeIndex) supaya .forecast() setelah .append()
# tahu persis "posisi berikutnya" -- index tanggal bursa kita bolong2 (libur/akhir
# pekan) jadi bikin statsmodels bingung soal frekuensi. Tanggal dipasang lagi nanti
# cuma utk keperluan visualisasi, bukan untuk pemodelan.
y_train = y_train.reset_index(drop=True)
X_train = X_train.reset_index(drop=True)
# Index test MELANJUTKAN dari akhir train (bukan reset ke 0 lagi) -- kalau reset
# ke 0, .append() nanti akan bingung dianggap data tumpang tindih dgn train.
idx_lanjut = range(len(y_train), len(y_train) + len(y_test))
y_test = pd.Series(y_test.values, index=idx_lanjut)
X_test = pd.DataFrame(X_test.values, index=idx_lanjut, columns=X_test.columns)
tanggal_test = test.index  # simpan terpisah utk pelaporan/plot

# SARIMAX/.append() bermasalah dgn DatetimeIndex trading-day (gap tak beraturan,
# tidak match freq pandas manapun) -- "ValueError: No supported index available"
# setelah append. Solusi: pakai RangeIndex polos utk internal model, tanggal asli
# cuma dipakai di luar model (plot, pelaporan).
y_train, X_train = y_train.reset_index(drop=True), X_train.reset_index(drop=True)
y_test, X_test = y_test.reset_index(drop=True), X_test.reset_index(drop=True)

# ============================================================
# Pemilihan order (p,d,q) via grid search AIC -- d=0 krn sudah stasioner
# ============================================================
print("=== Pemilihan order ARIMAX (grid search AIC, d=0) ===")
hasil_grid = []
for p in range(4):
    for q in range(3):
        try:
            m = SARIMAX(y_train, exog=X_train, order=(p, 0, q),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            hasil_grid.append({"p": p, "q": q, "aic": m.aic})
        except Exception:
            continue
grid_df = pd.DataFrame(hasil_grid).sort_values("aic")
print(grid_df.to_string(index=False))
best_p, best_q = int(grid_df.iloc[0]["p"]), int(grid_df.iloc[0]["q"])
print(f"\nOrder terpilih: ARIMAX({best_p},0,{best_q}), AIC={grid_df.iloc[0]['aic']:.2f}")

# ============================================================
# Fit final di train, lalu walk-forward: append data test satu-satu
# (update state via Kalman filter, TANPA refit ulang parameter)
# ============================================================
model = SARIMAX(y_train, exog=X_train, order=(best_p, 0, best_q),
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

pred = []
current = model
for i in range(len(test)):
    fc = current.forecast(steps=1, exog=X_test.iloc[[i]])
    pred.append(fc.iloc[0])
    current = current.append(y_test.iloc[[i]], exog=X_test.iloc[[i]], refit=False)
pred = np.array(pred)

joblib.dump(model, "outputs/models/arimax.joblib")

rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
dir_acc = (np.sign(pred) == np.sign(y_test)).mean()
print(f"\n=== EVALUASI ARIMAX({best_p},0,{best_q}) ===")
print(f"RMSE: {rmse:.6f}  MAE: {mae:.6f}  Directional Accuracy: {dir_acc:.4f}")

k = int((np.sign(pred) == np.sign(y_test.values)).sum())
res = binomtest(k, len(test), 0.5, alternative="greater")
print(f"Uji Binomial vs 50%: k={k}/{len(test)}, p={res.pvalue:.4f}",
      "-> SIGNIFIKAN" if res.pvalue < 0.05 else "-> tidak signifikan")

pd.DataFrame([{"model": "arimax", "order": f"({best_p},0,{best_q})", "rmse": rmse,
               "mae": mae, "directional_accuracy": dir_acc, "binom_p": res.pvalue}]
             ).to_csv("outputs/tables/evaluasi_arimax.csv", index=False)

# ============================================================
# Visualisasi: aktual vs prediksi
# ============================================================
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(tanggal_test, y_test, label="Aktual", lw=0.8, color="black")
ax.plot(tanggal_test, pred, label=f"Prediksi ARIMAX({best_p},0,{best_q})", lw=0.8, alpha=0.7)
ax.set_ylabel("Return IHSG")
ax.set_title("ARIMAX -- Aktual vs Prediksi (v1)")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/figures/arimax_aktual_vs_prediksi.png", dpi=150)
plt.close()
print("\nFigure & tabel tersimpan.")