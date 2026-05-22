# Eksperimen SML - Baharudin Yusuf

Proyek untuk **Submission Proyek Akhir - Membangun Sistem Machine Learning**

## 📋 Struktur Folder

```
Eksperimen_SML_Baharudin-Yusuf/
├── .github/workflows/
│   └── preprocessing.yml          # GitHub Actions Workflow (Advance)
├── namadataset_raw/               # Raw dataset folder
│   └── [dataset files]
├── preprocessing/
│   ├── Eksperimen_Baharudin-Yusuf.ipynb    # Notebook eksperimen (Basic)
│   ├── automate_Baharudin-Yusuf.py         # Automation script (Skilled)
│   └── namadataset_preprocessing/          # Preprocessed dataset
│       ├── preprocessed_data.csv
│       └── preprocessing_metadata.json
└── README.md
```

## 🎯 Kriteria Penilaian

### Basic (2 poin) ✓
- [x] Melakukan tahapan experimentation secara manual
- [x] Data loading pada notebook
- [x] EDA (Exploratory Data Analysis) pada notebook
- [x] Preprocessing pada notebook

**File**: `preprocessing/Eksperimen_Baharudin-Yusuf.ipynb`

### Skilled (3 poin) ✓
- [x] Tahap Basic terpenuhi
- [x] Membuat file `automate_Baharudin-Yusuf.py` dengan fungsi preprocessing otomatis
- [x] Konversi dari proses eksperimen dengan struktur yang berbeda

**File**: `preprocessing/automate_Baharudin-Yusuf.py`

### Advance (4 poin) ✓
- [x] Tahap Skilled terpenuhi
- [x] Workflow GitHub Actions untuk preprocessing otomatis
- [x] Preprocessing terpicu setiap ada perubahan dataset
- [x] Dataset terbaru diproses dan disimpan otomatis

**File**: `.github/workflows/preprocessing.yml`

## 🚀 Fitur GitHub Actions Workflow

### Triggers
Workflow dijalankan otomatis ketika:
1. **Push** ke branch `main` dengan perubahan di:
   - `namadataset_raw/` (folder raw dataset)
   - `preprocessing/automate_*.py` (script automation)

2. **Pull Request** dengan perubahan yang sama

3. **Manual trigger** (workflow_dispatch)

### Steps yang Dijalankan
1. **Checkout repository** - Mengambil kode terbaru
2. **Setup Python** - Persiapan environment (Python 3.10)
3. **Install dependencies** - Install library yang diperlukan
4. **Run preprocessing** - Jalankan `automate_Baharudin-Yusuf.py`
5. **Commit changes** - Simpan dataset yang sudah diproses
6. **Upload artifacts** - Simpan hasil sebagai artifact (retention 30 hari)
7. **Generate report** - Buat laporan preprocessing
8. **PR comment** - Berikan feedback di PR (jika ada)

## 📊 Proses Preprocessing Otomatis

### `automate_Baharudin-Yusuf.py`

Script ini mengotomatisasi seluruh proses preprocessing dengan class `DataPreprocessor`:

#### Tahapan Preprocessing:
1. **Data Loading** - Membaca dataset dari `namadataset_raw/`
2. **EDA** - Analisis data exploratory
3. **Handle Missing Values** - Menghapus baris dengan nilai kosong
4. **Remove Duplicates** - Menghapus duplikasi data
5. **Data Type Conversion** - Konversi tipe data
6. **Feature Engineering** - Engineering fitur (opsional)
7. **Normalization/Scaling** - Normalisasi data numerik menggunakan StandardScaler
8. **Save Data** - Simpan hasil ke `preprocessing/namadataset_preprocessing/`

### Output
- `preprocessed_data.csv` - Dataset yang sudah diproses
- `preprocessing_metadata.json` - Metadata proses preprocessing

## 🔧 Cara Menggunakan

### 1. Setup Dataset
Tempatkan dataset raw Anda di folder `namadataset_raw/`:
```bash
cp your_dataset.csv namadataset_raw/
```

### 2. Jalankan Preprocessing Manual
```bash
python preprocessing/automate_Baharudin-Yusuf.py
```

### 3. Jalankan Notebook Eksperimen
```bash
jupyter notebook preprocessing/Eksperimen_Baharudin-Yusuf.ipynb
```

### 4. Trigger GitHub Actions
Push perubahan ke `main` branch:
```bash
git add .
git commit -m "Add dataset for preprocessing"
git push origin main
```

Workflow akan berjalan otomatis dan:
- Memproses dataset
- Commit hasil ke repository
- Upload artifact untuk download

## 📥 Download Hasil

### Via GitHub UI
1. Buka repository di GitHub
2. Klik tab "Actions"
3. Pilih workflow "Automated Preprocessing Pipeline"
4. Klik completed run
5. Download "preprocessed-dataset" artifact

### Via GitHub CLI
```bash
gh run download <run_id> -n preprocessed-dataset
```

## 📝 Customization

### Ubah Nama Student
Ganti semua kemunculan `Baharudin-Yusuf` dengan nama Anda:
1. File notebook: `Eksperimen_[Nama-Anda].ipynb`
2. Script automation: `automate_[Nama-Anda].py`

### Ubah Nama Dataset
Sesuaikan nama folder di:
- `namadataset_raw/` → `[nama-dataset]_raw/`
- `preprocessing/namadataset_preprocessing/` → `preprocessing/[nama-dataset]_preprocessing/`

### Tambah Custom Preprocessing Steps
Edit class `DataPreprocessor` di `automate_Baharudin-Yusuf.py`:
```python
def custom_processing(self):
    """Tambah step custom preprocessing"""
    print("[CUSTOM] Custom processing...")
    # Tambah logika Anda di sini
    print("✓ Custom processing completed")

# Tambah ke method preprocess()
self.custom_processing()
```

## 🔐 Repository Settings

Pastikan setup repository:
1. Actions enabled ✓
2. Workflow permissions: "Read and write permissions" ✓
3. Auto-merge: Optional sesuai kebutuhan

## 📚 Dependencies

- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning (untuk StandardScaler)
- `jupyter` - Jupyter notebook
- `matplotlib`, `seaborn` - Visualization (untuk notebook)

## ⚠️ Notes

- Dataset besar dapat memperlambat workflow execution
- Artifact otomatis dihapus setelah 30 hari
- Commit otomatis hanya dilakukan jika ada perubahan data
- Jika error, lihat workflow logs di GitHub Actions

## 📞 Troubleshooting

### Workflow gagal dengan error "No data files found"
**Solusi**: Pastikan ada file dataset di folder `namadataset_raw/`

### Data tidak ter-commit otomatis
**Solusi**: Periksa branch protection rules, pastikan tidak menghalangi push otomatis

### Artifact tidak ada
**Solusi**: Lihat logs di GitHub Actions untuk error detail, pastikan script berjalan sukses

## ✅ Checklist Submission

- [x] Folder structure sesuai kriteria
- [x] Notebook eksperimen dengan semua tahapan
- [x] Script automation untuk preprocessing
- [x] GitHub Actions workflow untuk automated processing
- [x] Dataset preprocessed tersimpan di repository
- [x] README dengan dokumentasi lengkap

---

**Status**: ✅ **ADVANCE LEVEL (4 poin)**

Last updated: 2026-05-22
