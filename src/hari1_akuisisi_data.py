import glob
import os
import sys
from pathlib import Path
import pandas as pd
import yfinance as yf
from fredapi import Fred

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FRED_API_KEY

# INDONIA baru mulai dihitung 2 Januari 2019 (dikonfirmasi dari situs BI --
# bukan Agustus 2018 seperti dugaan awal). Raw akuisisi ikut mulai dari sini,
# supaya tidak ada buffer yang terbuang percuma (variabel lain toh akan
# dropna() menunggu INDONIA tersedia juga).
START, END = "2019-01-02", "2026-06-30"
OUT = "data_raw"
os.makedirs(OUT, exist_ok=True)

# ============================================================
# PREFLIGHT CHECK -- cek SEMUA file manual sebelum mulai proses apapun,
# supaya kalau ada yang belum diunduh, ketahuan sekali di awal (bukan
# berhenti satu-satu tiap kali dijalankan ulang seperti kemarin).
# ============================================================
file_wajib = ["epu_daily_raw.csv", "bi_rate_raw.csv", "indonia_raw.csv", "inflasi_raw.csv"]
hilang = [f for f in file_wajib if not Path(f"{OUT}/{f}").exists()]
pdb_files = sorted(glob.glob(f"{OUT}/pdb_raw_*.csv"))

if hilang or not pdb_files:
    print("BERHENTI -- file mentah berikut belum ada di data_raw/, unduh dulu:")
    for f in hilang:
        print(f"  - {f}")
    if not pdb_files:
        print("  - pdb_raw_<tahun>.csv (minimal 1 file, pola nama: pdb_raw_2019.csv, pdb_raw_2020.csv, dst)")
    sys.exit(1)

print(f"Preflight OK -- semua file manual ada. PDB: {len(pdb_files)} file tahun ditemukan.")

# --- Yahoo Finance: 6 ticker sekali download ---
tickers = {"^JKSE": "ihsg", "USDIDR=X": "usdidr", "^GSPC": "sp500",
           "CL=F": "oil_wti", "DX-Y.NYB": "dxy", "^VIX": "vix"}
yf_data = yf.download(list(tickers), start=START, end=END, auto_adjust=True)["Close"]
yf_data = yf_data.rename(columns=tickers)
for col in yf_data.columns:
    yf_data[col].dropna().to_csv(f"{OUT}/{col}.csv")

# --- FRED: 2 series harian (DFF, bukan FEDFUNDS yang bulanan) ---
fred = Fred(api_key=FRED_API_KEY)
for series_id, name in {"DFF": "fed_funds_rate", "DGS10": "ust10y"}.items():
    s = fred.get_series(series_id, observation_start=START, observation_end=END).rename(name)
    s = s.ffill()  # isi gap hari libur finansial AS yang bukan weekend
    s.to_csv(f"{OUT}/{name}.csv")

# --- Tanggal campuran (Excel kadang parse otomatis, kadang enggak) ---
BULAN_ID = {"januari":1,"jan":1,"februari":2,"feb":2,"maret":3,"mar":3,"april":4,"apr":4,
            "mei":5,"juni":6,"jun":6,"juli":7,"jul":7,"agustus":8,"agu":8,"agt":8,
            "september":9,"sep":9,"oktober":10,"okt":10,"november":11,"nov":11,"desember":12,"des":12}

def parse_tanggal(series):
    hasil = pd.to_datetime(series, format="mixed", dayfirst=True, errors="coerce")
    for i in hasil[hasil.isna()].index:
        d, b, y = str(series[i]).split()
        hasil[i] = pd.Timestamp(int(y), BULAN_ID[b.lower()], int(d))
    return hasil

def clean_cols(df):
    df.columns = df.columns.str.replace("<br>", " ", regex=False).str.strip()
    return df

assert parse_tanggal(pd.Series(["23-Jun-26", "29 Mei 2026"]))[1] == pd.Timestamp(2026, 5, 29)

# --- EPU harian ---
epu_raw = clean_cols(pd.read_csv(f"{OUT}/epu_daily_raw.csv"))
epu = pd.Series(epu_raw["daily_policy_index"].values,
                 index=pd.to_datetime(epu_raw[["year", "month", "day"]]),
                 name="epu").sort_index().loc[START:END]
epu.to_csv(f"{OUT}/epu_daily.csv")

# --- BI Rate ---
bi = clean_cols(pd.read_csv(f"{OUT}/bi_rate_raw.csv", sep=";"))
bi["Tanggal"] = parse_tanggal(bi["Tanggal"])
bi_rate = bi.dropna(subset=["Tanggal"]).assign(
    bi_rate=lambda d: d["BI-7Day-RR"].astype(str).str.replace("%", "").str.strip().astype(float)
).set_index("Tanggal")["bi_rate"].sort_index().loc[START:END]
bi_rate.to_csv(f"{OUT}/bi_rate.csv")

# --- INDONIA ---
ind = clean_cols(pd.read_csv(f"{OUT}/indonia_raw.csv", sep=";"))
ind["Tanggal Publikasi"] = parse_tanggal(ind["Tanggal Publikasi"])
indonia = ind.dropna(subset=["Tanggal Publikasi"]).assign(
    indonia=lambda d: d["IndONIA (%)"].astype(str).str.replace(",", ".").astype(float)
).set_index("Tanggal Publikasi")["indonia"].sort_index().loc[START:END]
indonia.to_csv(f"{OUT}/indonia.csv")

# ============================================================
# Inflasi (YoY, format long "Periode"/"Data Inflasi") & PDB (format lebar
# BPS per-sektor, kita ambil baris agregat "PRODUK DOMESTIK BRUTO" saja) --
# forward-fill dari TANGGAL RILIS RESMI (bukan tanggal periode yang
# dilaporkan), offset konservatif supaya tidak look-ahead bias.
# Pola riil terverifikasi: inflasi rilis hari kerja pertama bulan
# berikutnya (dipakai offset hari-5); PDB rilis ~35-36 hari setelah
# triwulan berakhir (dipakai offset hari-40).
# ============================================================
import re

def rilis_dari_periode_bulanan(tahun, bulan, hari_setelah_bulan_berikutnya):
    awal_bulan_berikutnya = pd.Timestamp(tahun, bulan, 1) + pd.DateOffset(months=1)
    return awal_bulan_berikutnya + pd.Timedelta(days=hari_setelah_bulan_berikutnya - 1)

def rilis_dari_triwulan(tahun, triwulan, hari_setelah_triwulan_berakhir):
    akhir_bulan_tw = {1: 3, 2: 6, 3: 9, 4: 12}[triwulan]
    akhir_triwulan = pd.Timestamp(tahun, akhir_bulan_tw, 1) + pd.offsets.MonthEnd(0)
    return akhir_triwulan + pd.Timedelta(days=hari_setelah_triwulan_berakhir)

# --- Inflasi YoY: kolom "Periode" ("Juni 2026") + "Data Inflasi" ("3.34 %") ---
inf_raw = pd.read_csv(f"{OUT}/inflasi_raw.csv", sep=None, engine="python")
inf_raw.columns = inf_raw.columns.str.strip()
bulan_tahun = inf_raw["Periode"].str.extract(r"(?P<bulan>\w+)\s+(?P<tahun>\d{4})")
inf_raw["bulan_num"] = bulan_tahun["bulan"].str.lower().map(BULAN_ID)
inf_raw["tahun"] = bulan_tahun["tahun"].astype(int)
assert inf_raw["bulan_num"].isna().sum() == 0, "Ada nama bulan yang tidak dikenali -- cek isi kolom Periode"
inf_raw["tanggal_rilis"] = inf_raw.apply(
    lambda r: rilis_dari_periode_bulanan(r["tahun"], r["bulan_num"], 5), axis=1)
inf_raw["nilai"] = inf_raw["Data Inflasi"].astype(str).str.replace("%", "").str.strip().astype(float)
inflasi = inf_raw.set_index("tanggal_rilis")["nilai"].sort_index().loc[START:END]
inflasi.name = "inflasi"
inflasi.to_csv(f"{OUT}/inflasi.csv")
print(f"inflasi: {len(inflasi)} observasi, rilis {inflasi.index.min()} s/d {inflasi.index.max()}")

# --- PDB: satu file per tahun, ambil baris "PRODUK DOMESTIK BRUTO",
# kolom y-on-y (indeks 11-14 = TW I, II, III, IV -- lihat struktur asli
# tabel BPS "[Seri 2010] Laju Pertumbuhan PDB", 3 blok 5-kolom:
# c-to-c, q-to-q, y-on-y, masing2 TW I-IV + Tahunan).
pdb_rows = []
for f in pdb_files:
    tahun_match = re.search(r"(20\d{2})", Path(f).stem)
    assert tahun_match, f"Tidak nemu tahun 4-digit di nama file {f} -- rename filenya"
    tahun = int(tahun_match.group(1))
    raw = pd.read_csv(f, header=None)
    baris_pdb = raw[raw[0].astype(str).str.contains("PRODUK DOMESTIK BRUTO", na=False)]
    assert len(baris_pdb) == 1, f"Baris 'PRODUK DOMESTIK BRUTO' tidak ketemu/ganda di {f}"
    yoy = pd.to_numeric(baris_pdb.iloc[0, 11:15], errors="coerce").values  # TW I,II,III,IV; "-" -> NaN
    for tw, nilai in zip([1, 2, 3, 4], yoy):
        if pd.notna(nilai):  # skip triwulan yang belum dirilis (biasanya tahun berjalan)
            pdb_rows.append({"tanggal_rilis": rilis_dari_triwulan(tahun, tw, 40), "pdb": nilai})

pdb_df = pd.DataFrame(pdb_rows).set_index("tanggal_rilis").sort_index()
assert pdb_df["pdb"].between(-20, 20).all(), "Ada nilai PDB di luar -20%..20% -- kemungkinan salah kolom, cek manual"
pdb = pdb_df["pdb"].loc[START:END]
pdb.to_csv(f"{OUT}/pdb.csv")
print(f"pdb: {len(pdb)} observasi (dari {len(pdb_files)} file tahun), rilis {pdb.index.min()} s/d {pdb.index.max()}")

print({"ihsg": len(yf_data["ihsg"].dropna()), "epu": len(epu),
       "bi_rate": len(bi_rate), "indonia": len(indonia),
       "inflasi": len(inflasi), "pdb": len(pdb)})
print(f"\nCEK MANUAL: sesuaikan nama kolom di inflasi_raw.csv/pdb_raw_<tahun>.csv "
      f"dengan format asli hasil unduhan Anda kalau beda dari asumsi di komentar kode.")