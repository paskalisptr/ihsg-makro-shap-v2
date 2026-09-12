import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

OUT = "data_raw"
# Dataset_final.csv disimpan MASIH dengan buffer penuh (dari 2019-01-02) --
# sama seperti pipeline v1 (buffer 2020 tersimpan, difilter ke 2021+ belakangan
# di hari3_eda.py/hari4_SHAP.py). Filter periode analisis final (2019-09-01)
# dilakukan di skrip HILIR, bukan di sini -- konsisten dengan konvensi lama.

ihsg = pd.read_csv(f"{OUT}/ihsg.csv", index_col=0, parse_dates=True)["ihsg"]
df = pd.DataFrame({"ihsg": ihsg})

# variabel yang sudah harian (kalender/bursa) -- reindex ke hari bursa IHSG, ffill dulu baru potong
daily_vars = ["usdidr", "sp500", "oil_wti", "dxy", "vix", "ust10y", "fed_funds_rate", "epu_daily", "indonia"]
for name in daily_vars:
    s = pd.read_csv(f"{OUT}/{name}.csv", index_col=0, parse_dates=True).iloc[:, 0]
    col = name.replace("_daily", "")
    df[col] = s.reindex(df.index.union(s.index)).ffill().reindex(df.index)

# variabel event/rilis (berubah jarang -- RDG, rilis bulanan/triwulanan):
# bi_rate, dan BARU inflasi + PDB. Pola sama persis (ffill ke kalender penuh
# dulu, baru potong ke hari bursa) -- ditulis sekali sebagai fungsi, dipakai
# 3x, bukan copy-paste blok yang sama tiga kali.
def gabung_event(nama_file):
    s = pd.read_csv(f"{OUT}/{nama_file}.csv", index_col=0, parse_dates=True).iloc[:, 0]
    s_full = s.reindex(pd.date_range(s.index.min(), df.index.max())).ffill()
    return s_full.reindex(df.index)

for nama_file in ["bi_rate", "inflasi", "pdb"]:
    df[nama_file] = gabung_event(nama_file)

df = df.dropna()  # buang baris awal sebelum SEMUA variabel (termasuk inflasi/PDB) punya nilai
assert df.isna().sum().sum() == 0

# return log IHSG + lag 1 hari semua fitur makro
df["return_ihsg"] = np.log(df["ihsg"] / df["ihsg"].shift(1))
fitur = [c for c in df.columns if c not in ("ihsg", "return_ihsg")]
df_model = df[fitur].shift(1).join(df["return_ihsg"]).dropna()
df_model.columns = [c if c == "return_ihsg" else f"{c}_lag1" for c in df_model.columns]

df_model.to_csv("data_processed/dataset_final.csv")
print(df_model.shape, df_model.index.min(), "s/d", df_model.index.max())

for c in df_model.columns:
    p = adfuller(df_model[c].dropna())[1]
    print(f"{c:20s} ADF p={p:.4f}", "stasioner" if p < 0.05 else "TIDAK stasioner -> perlu diff")

# inflasi & PDB ditambahkan ke daftar differencing (perlakuan sama seperti
# variabel level makro lain) -- ADF di bawah akan konfirmasi/bantah empiris,
# sama seperti VIX yang ternyata TIDAK perlu diff meski awalnya diduga perlu.
non_stationary = ["usdidr", "sp500", "oil_wti", "dxy", "ust10y", "fed_funds_rate",
                   "epu", "indonia", "bi_rate", "inflasi", "pdb"]
for col in non_stationary:
    df_model[f"{col}_lag1"] = df_model[f"{col}_lag1"].diff()
df_model = df_model.dropna()

# validasi ulang -- pastikan diff beneran bikin stasioner, bukan asal transform
for c in df_model.columns:
    p = adfuller(df_model[c].dropna())[1]
    tag = "stasioner" if p < 0.05 else "MASIH TIDAK stasioner -- perlu dicek manual"
    print(f"{c:20s} ADF p={p:.4f} {tag}")

df_model.to_csv("data_processed/dataset_final.csv")
print(df_model.shape)