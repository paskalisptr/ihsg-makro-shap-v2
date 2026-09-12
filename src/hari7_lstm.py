import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import binomtest
import tensorflow as tf
from tensorflow.keras import Sequential, layers, callbacks

Path("outputs/figures").mkdir(parents=True, exist_ok=True)
Path("outputs/tables").mkdir(parents=True, exist_ok=True)
Path("outputs/models").mkdir(parents=True, exist_ok=True)
tf.random.set_seed(42)

LOOKBACK = 20  # hari -- konsisten dgn window rolling volatility yg sudah dipakai di tempat lain

df = pd.read_csv("data_processed/dataset_final.csv", index_col=0, parse_dates=True).loc["2019-09-01":]
y, X = df["return_ihsg"], df.drop(columns="return_ihsg")
train, test = df.loc[:"2024-12-31"], df.loc["2024-12-31":].iloc[1:]

# Scaler di-fit HANYA di train -- cegah leakage informasi skala dari test
scaler = StandardScaler().fit(train[X.columns])
X_scaled = pd.DataFrame(scaler.transform(df[X.columns]), index=df.index, columns=X.columns)


def buat_sequence(X_scaled_df, y_series, indeks_target):
    """Untuk tiap tanggal di indeks_target, ambil LOOKBACK hari fitur SEBELUM
    tanggal itu (termasuk hari itu sendiri, karena fitur sudah _lag1 -- aman)."""
    Xs, ys = [], []
    pos_map = {d: i for i, d in enumerate(X_scaled_df.index)}
    for tgl in indeks_target:
        pos = pos_map[tgl]
        if pos - LOOKBACK + 1 < 0:
            continue
        Xs.append(X_scaled_df.iloc[pos - LOOKBACK + 1: pos + 1].values)
        ys.append(y_series.loc[tgl])
    return np.array(Xs), np.array(ys)


X_train_seq, y_train_seq = buat_sequence(X_scaled, y, train.index)
X_test_seq, y_test_seq = buat_sequence(X_scaled, y, test.index)
print(f"Train sequences: {X_train_seq.shape}, Test sequences: {X_test_seq.shape}")

# Validasi internal 15% akhir train (time-based, bukan acak) utk early stopping
n_val = int(len(X_train_seq) * 0.15)
X_tr, y_tr = X_train_seq[:-n_val], y_train_seq[:-n_val]
X_val, y_val = X_train_seq[-n_val:], y_train_seq[-n_val:]

# ============================================================
# Arsitektur SENGAJA sederhana: 1 layer LSTM kecil + dropout + dense
# ============================================================
model = Sequential([
    layers.Input(shape=(LOOKBACK, X.shape[1])),
    layers.LSTM(16),
    layers.Dropout(0.2),
    layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")
es = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=100,
                  batch_size=32, callbacks=[es], verbose=0)
print(f"Training berhenti di epoch {len(hist.history['loss'])} (early stopping)")

model.save("outputs/models/lstm.keras")

pred = model.predict(X_test_seq, verbose=0).flatten()
rmse = np.sqrt(mean_squared_error(y_test_seq, pred))
mae = mean_absolute_error(y_test_seq, pred)
dir_acc = (np.sign(pred) == np.sign(y_test_seq)).mean()
print(f"\n=== EVALUASI LSTM ===")
print(f"RMSE: {rmse:.6f}  MAE: {mae:.6f}  Directional Accuracy: {dir_acc:.4f}")

k = int((np.sign(pred) == np.sign(y_test_seq)).sum())
res = binomtest(k, len(y_test_seq), 0.5, alternative="greater")
print(f"Uji Binomial vs 50%: k={k}/{len(y_test_seq)}, p={res.pvalue:.4f}",
      "-> SIGNIFIKAN" if res.pvalue < 0.05 else "-> tidak signifikan")

pd.DataFrame([{"model": "lstm", "lookback": LOOKBACK, "rmse": rmse, "mae": mae,
               "directional_accuracy": dir_acc, "binom_p": res.pvalue}]
             ).to_csv("outputs/tables/evaluasi_lstm.csv", index=False)

# ============================================================
# Visualisasi: loss curve (bukti tidak overfit parah) + aktual vs prediksi
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(hist.history["loss"], label="Train loss")
axes[0].plot(hist.history["val_loss"], label="Val loss")
axes[0].set_title("Kurva Training LSTM")
axes[0].legend()

tgl_test = test.index[-len(y_test_seq):]
axes[1].plot(tgl_test, y_test_seq, label="Aktual", lw=0.8, color="black")
axes[1].plot(tgl_test, pred, label="Prediksi LSTM", lw=0.8, alpha=0.7)
axes[1].set_title("LSTM -- Aktual vs Prediksi (v1)")
axes[1].legend()
plt.tight_layout()
plt.savefig("outputs/figures/lstm_hasil.png", dpi=150)
plt.close()
print("\nFigure & tabel tersimpan.")