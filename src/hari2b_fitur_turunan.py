import pandas as pd

df = pd.read_csv("data_processed/dataset_final.csv", index_col=0, parse_dates=True)

# Momentum: return_ihsg hari-hari sebelumnya (autoregresif) -- shift dari
# return_ihsg yang SUDAH ADA di dataset (bukan re-derive dari ihsg mentah,
# supaya index/urutan tetap konsisten dengan hari2_gabung_data.py).
for lag in [1, 2, 3]:
    df[f"return_ihsg_lag{lag}"] = df["return_ihsg"].shift(lag)

# Rolling volatility -- WAJIB shift(1): rolling(window).std() secara default
# ikut menghitung hari ini sendiri, padahal return_ihsg hari ini adalah target
# yang mau diprediksi. shift(1) memastikan volatilitas yang dipakai sebagai
# fitur murni dari HARI-HARI SEBELUM hari yang diprediksi.
for window in [5, 20]:
    df[f"vol_roll{window}_lag1"] = df["return_ihsg"].rolling(window).std().shift(1)

df = df.dropna()  # buang baris awal yang belum punya cukup histori utk lag3/rolling20
df.to_csv("data_processed/dataset_final.csv")

fitur_baru = [c for c in df.columns if "return_ihsg_lag" in c or "vol_roll" in c]
print(f"Shape akhir: {df.shape}, periode {df.index.min()} s/d {df.index.max()}")
print(f"Fitur baru ({len(fitur_baru)}): {fitur_baru}")
print(f"Total fitur sekarang: {len(df.columns) - 1} (di luar return_ihsg)")