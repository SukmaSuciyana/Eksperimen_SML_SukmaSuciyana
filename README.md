# Eksperimen Machine Learning - Predictive Maintenance Pipeline

Repositori ini berisi eksperimen machine learning untuk predictive maintenance pada sistem pipa minyak dan gas. Dataset yang digunakan fokus pada prediksi kondisi pipa berdasarkan berbagai parameter operasional.

## 📊 Dataset

Dataset yang digunakan berasal dari [Kaggle - Predictive Maintenance Oil and Gas Pipeline](https://www.kaggle.com/datasets/muhammadwaqas023/predictive-maintenance-oil-and-gas-pipeline-data/data)

**Fitur-fitur Dataset:**
- **Pipe_Size_mm**: Diameter pipa dalam milimeter
- **Thickness_mm**: Ketebalan dinding pipa
- **Material**: Jenis material pipa (Carbon Steel, PVC, HDPE)
- **Grade**: Grade material (ASTM A333 Grade 6, ASTM A106 Grade B, API 5L X52, dll)
- **Max_Pressure_psi**: Tekanan maksimum yang dialami pipa (psi)
- **Temperature_C**: Suhu fluida di dalam pipa (°C)
- **Corrosion_Impact_Percent**: Persentase dampak korosi
- **Thickness_Loss_mm**: Kehilangan ketebalan akibat aus atau korosi
- **Material_Loss_Percent**: Persentase kehilangan material
- **Time_Years**: Umur pipa dalam tahun
- **Condition**: Target - Kondisi operasional (Normal, Moderate, Critical)

## 🗂️ Struktur Folder

```
Eksperimen_SML_sukmasucii/
│
├── dataset_raw/                              # Dataset mentah
│   └── market_pipe_thickness_loss_dataset.csv
│
├── preprocessing/                            # File preprocessing
│   ├── automate_sukmasuci.py                # Script otomasi preprocessing
│   ├── eksperimen_MSML_sukmasuci.ipynb      # Notebook eksplorasi dan preprocessing
│   ├── preprocessed_data_auto_*/            # Hasil preprocessing otomatis
│   └── preprocessed_data_manual/            # Hasil preprocessing manual
│
└── README.md                                 # File ini
```

## 🚀 Cara Menggunakan

### 1. Setup Environment

```bash
# Buat conda environment baru
conda create -n MSML python=3.10
conda activate MSML

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### 2. Preprocessing Data

Ada dua cara untuk melakukan preprocessing:

#### A. Menggunakan Script Otomatis

```bash
# Jalankan dari root folder
python preprocessing/automate_sukmasuci.py
```

Script ini akan:
- Load dataset dari `dataset_raw/`
- Melakukan preprocessing otomatis menggunakan sklearn Pipeline
- Menyimpan hasil preprocessing dengan timestamp

#### B. Menggunakan Jupyter Notebook

```bash
# Buka notebook untuk eksplorasi interaktif
jupyter notebook preprocessing/eksperimen_MSML_sukmasuci.ipynb
```

Notebook ini berisi:
- Exploratory Data Analysis (EDA) lengkap
- Visualisasi distribusi data
- Preprocessing step-by-step
- Analisis korelasi dan statistik

## 🔍 Tahapan Preprocessing

1. **Load Data**: Membaca dataset dari file CSV
2. **Drop Redundant Features**: Menghapus fitur `Pipe_Size_mm` yang redundan dengan `Thickness_mm`
3. **Feature Transformation**: Transformasi log untuk fitur yang skewed (`Thickness_mm`, `Material_Loss_Percent`)
4. **Data Splitting**: Split data menjadi train (80%) dan test (20%) dengan stratified sampling
5. **Scaling**: MinMax scaling untuk fitur numerik (range 0-1)
6. **Encoding**: 
   - One-Hot Encoding untuk fitur kategoris (`Material`, `Grade`)
   - Label Encoding untuk target (`Condition`)
7. **Save Results**: Menyimpan data yang sudah diproses dan pipeline untuk keperluan modeling

## 📈 Hasil Preprocessing

Setelah preprocessing, data akan memiliki:
- **17 fitur** (setelah one-hot encoding)
- **Range nilai**: 0-1 (setelah scaling)
- **Target encoding**: 
  - Normal = 2
  - Moderate = 1
  - Critical = 0

Output yang tersimpan:
- `X_train.csv` - Data training features
- `X_test.csv` - Data testing features
- `y_train.csv` - Data training target
- `y_test.csv` - Data testing target
- `feature_names.csv` - Nama-nama fitur hasil transformasi
- `pipeline.joblib` - Pipeline preprocessing untuk reuse
- `pipeline_label_encoder.joblib` - Label encoder untuk target

## 📊 Exploratory Data Analysis

Notebook berisi analisis mendalam:
- ✅ Statistik deskriptif data
- ✅ Distribusi fitur numerik dan kategoris
- ✅ Deteksi outlier dengan IQR method
- ✅ Analisis korelasi antar fitur
- ✅ Visualisasi hubungan fitur dengan target
- ✅ Pairplot untuk fitur-fitur penting
- ✅ Violin plot untuk distribusi detail
- ✅ Heatmap rata-rata fitur per kondisi

## 💡 Insight dari EDA

1. **Multikolinearitas**: `Thickness_mm` dan `Pipe_Size_mm` sangat berkorelasi tinggi, sehingga salah satu dihapus
2. **Fitur Penting**: `Thickness_Loss_mm` dan `Material_Loss_Percent` menunjukkan perbedaan signifikan antar kelas
3. **Distribusi Target**: Data sedikit imbalanced dengan dominasi kelas Critical
4. **Outliers**: Beberapa outlier terdeteksi tapi masih dalam range wajar, diatasi dengan transformasi log

## 🛠️ Teknologi yang Digunakan

- **Python 3.10+**
- **pandas** - Manipulasi data
- **numpy** - Operasi numerik
- **scikit-learn** - Machine learning & preprocessing
- **matplotlib & seaborn** - Visualisasi
- **joblib** - Serialisasi model/pipeline

---

**Happy Coding!** 🎉
