# ihsg-makro-shap 2.0

Pipeline eksperimen lanjutan: perbandingan pengaruh faktor domestik vs global
terhadap pergerakan IHSG, dengan interpretabilitas SHAP.

Repo ini **terpisah dari pipeline utama yang sudah settled** (2021-2026,
dipakai di BAB 1-3 & paper JAIC). Tujuan repo ini murni eksperimental:
menguji apakah perluasan periode + variabel makro tambahan bisa menaikkan
akurasi/menurunkan RMSE tanpa data leakage. Kalau hasilnya tidak lebih baik,
pipeline lama tetap jadi hasil final skripsi.

## Struktur folder

```
src/                    skrip pipeline, urut sesuai nama file (hari1 -> hari2 -> ...)
data_raw/                data mentah per sumber (hasil hari1, tidak di-commit)
data_processed/          dataset_final.csv gabungan (hasil hari2, tidak di-commit)
notebooks/                eksplorasi ad-hoc, opsional
outputs/figures/          semua .png
outputs/tables/            semua .csv hasil (SHAP, ablation, uji statistik, dst)
outputs/models/            model regresi tersimpan (.joblib)
outputs/models_klasifikasi/ model klasifikasi tersimpan, kalau dipakai
```

## Cara menjalankan

1. Clone repo, `pip install -r requirements.txt`
2. Copy `config_template.py` -> `config.py`, isi `FRED_API_KEY` (daftar gratis di https://fred.stlouisfed.org/docs/api/api_key.html)
3. Unduh manual ke `data_raw/` (BPS/BI tidak punya API publik sederhana):
   - `bi_rate_raw.csv`, `indonia_raw.csv` -- dari bi.go.id
   - `epu_daily_raw.csv` -- dari policyuncertainty.com
   - `inflasi_raw.csv`, `pdb_raw.csv`, `cci_raw.csv` -- dari bps.go.id / bi.go.id
     (cek komentar di `hari1_akuisisi_data.py` untuk format kolom yang diharapkan --
     SESUAIKAN dengan format asli hasil unduhan, jangan asumsikan langsung cocok)
4. Jalankan skrip berurutan sesuai nomor nama file di `src/`

## Status pipeline (v2.0, per eksperimen backlog)

- [x] `hari1_akuisisi_data.py` -- periode diperlebar ke 2018-09-01, + inflasi/PDB/CCI
      (forward-fill dari tanggal rilis resmi, offset konservatif -- lihat komentar kode)
- [ ] `hari2_gabung_data.py` -- BELUM diupdate untuk 3 variabel baru (masih versi lama,
      9 fitur). Perlu revisi sebelum dipakai.
- [x] `hari3_eda.py` -- ditambah visualisasi outlier bertanda, boxplot per fitur,
      time series per fitur, distribusi train-vs-test
- [ ] `hari3_pemodelan.py` -- belum direview/direvisi untuk fitur turunan
- [ ] Fitur turunan (momentum, rolling volatility) -- belum ditulis

## Perbedaan dari pipeline v1 (settled)

| | v1 (settled, dipakai di BAB 1-3) | v2.0 (eksperimen, repo ini) |
|---|---|---|
| Periode | 2021-01-01 s/d 2026-06-30 | 2019-09-01 s/d 2026-06-30 |
| Fitur | 10 (3 domestik + 7 global) | 13 (rencana: +inflasi, +PDB, +CCI) |
| Fitur turunan | Tidak ada | Direncanakan (momentum, rolling vol) |
| Status | Selesai, teraudit, hasil settled | Berjalan, belum ada hasil final |
