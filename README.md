# Reproduksi — Application of MobileNetV2 to Waste Classification

Reproduksi paper:

> Yong, L., Ma, L., Sun, D., & Du, L. (2023). **Application of MobileNetV2 to waste
> classification.** *PLOS ONE*, 18(3), e0282336.
> https://doi.org/10.1371/journal.pone.0282336

Kode asli paper ada di folder `Source Code/`. Folder ini berisi notebook yang menulis ulang
pipeline klasifikasi sampah: MobileNetV2 (PyTorch, transfer learning) dan CNN pembanding
(Keras di kode asli, ditiru dengan PyTorch di sini), mulai dari pencarian dataset, EDA,
split, training, evaluasi, analisis kesalahan, sampai inferensi.

**Ringkasan:** notebook memakai dataset Kaggle 30 kelas (15.000 file dari group `default` dan
`real_world`), menghapus duplikat exact dengan MD5 (sisa 11.263 gambar unik), membagi data
8:1:1 secara stratified (9.006 train / 1.124 validation / 1.133 test), lalu melatih
MobileNetV2 dan CNN pembanding. Evaluasi utama mengikuti `test_model.py` pada GitHub: test
accuracy, cross-entropy loss, dan `classification_report` untuk 3 superclass. Dataset asli
paper (245 kelas) tidak tersedia, dan dataset ini tidak memiliki kelas hazardous.

## Struktur folder

```text
checkpoint_1/
├── README.md                  ← dokumen ini
├── Reproduce Code/
│   ├── Reproduce_MobileNetV2_Waste_Classification.ipynb   ← notebook utama (deliverable)
│   ├── README.md              ← detail teknis notebook (setup, output, smoke test)
│   ├── requirements.txt
│   └── outputs/               ← hasil run (dibuat otomatis, tidak di-commit)
├── Source Code/               ← kode asli dari GitHub paper
│   ├── train_245_class.py     ← training MobileNetV2 (PyTorch)
│   ├── train_cnn.py           ← CNN pembanding (TensorFlow/Keras)
│   ├── test_model.py          ← evaluasi (test accuracy, loss, classification report)
│   ├── utils.py, test4singleimg.py, test4dataset.py, test_model_acc.py
│   ├── main.py                ← Flask backend
│   ├── window_trash.py        ← GUI PyQt
│   ├── imgs/
│   └── WeChat applet/         ← mini program WeChat
├── Datasets/
│   └── images/images/         ← 30 kelas × {default, real_world}
└── Paper/
    └── journal.pone.0282336.pdf
```

## Menjalankan notebook

Notebook `Reproduce Code/Reproduce_MobileNetV2_Waste_Classification.ipynb` berjalan di
**Kaggle**, **Google Colab**, dan **lokal**.

### Kaggle (direkomendasikan)

1. Buat notebook baru, attach dataset lewat **Add Data → Datasets** dengan nama
   `recyclable-and-household-waste-classification`.
2. Upload notebook ke `Reproduce Code/`.
3. Aktifkan GPU accelerator (opsional, tapi jauh lebih cepat).
4. **Run All**. Dataset terdeteksi otomatis, tidak ada proses download.

### Google Colab

1. Upload notebook ke `Reproduce Code/`.
2. **Runtime → Change runtime type → GPU** (opsional).
3. **Run All**. Dataset diunduh otomatis via `kagglehub`; jika Kaggle meminta autentikasi,
   tambahkan secret `KAGGLE_API_TOKEN`.
4. Alternatif: upload dataset manual lalu set `TRASH_DATA_DIR` di Bagian 1.

### Lokal

```bash
cd "Reproduce Code"
python -m pip install -r requirements.txt
jupyter notebook Reproduce_MobileNetV2_Waste_Classification.ipynb
```

Tidak wajib: `requirements.txt` opsional karena notebook memasang library yang kurang sendiri
di cell pertama Bagian 1. `pip install -r requirements.txt` tetap disarankan agar environment
dapat direproduksi persis.

Path dataset lokal yang dipakai otomatis: `../Datasets/images/images/`. Kalau dataset berada
di tempat lain, set environment variable:

```bash
export TRASH_DATA_DIR="/absolute/path/to/Datasets/images/images"
```

## Dataset

**Kaggle:** [Recyclable and Household Waste Classification](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification)
(Alistair King, 2024). 30 kelas, 15.000 file PNG, tiap kelas punya group `default` dan
`real_world`.

| Tahap | Jumlah |
|---|---:|
| File ditemukan | 15.000 |
| Duplikat exact dihapus (MD5) | 3.737 |
| Gambar unik | 11.263 |
| Train / Validation / Test | 9.006 / 1.124 / 1.133 |

Duplikat dihapus **sebelum** split, lalu dicek ulang agar tidak ada MD5 yang sama muncul di
split berbeda. Group `default` (7.019) dan `real_world` (4.244) keduanya dipakai, dan
split dibuat stratified per (kelas, source group) dengan `SEED = 42`.

Kelas: `aerosol_cans`, `aluminum_food_cans`, `aluminum_soda_cans`, `cardboard_boxes`,
`cardboard_packaging`, `clothing`, `coffee_grounds`, `disposable_plastic_cutlery`, `eggshells`,
`food_waste`, `glass_beverage_bottles`, `glass_cosmetic_containers`, `glass_food_jars`,
`magazines`, `newspaper`, `office_paper`, `paper_cups`, `plastic_cup_lids`,
`plastic_detergent_bottles`, `plastic_food_containers`, `plastic_shopping_bags`,
`plastic_soda_bottles`, `plastic_straws`, `plastic_trash_bags`, `plastic_water_bottles`,
`shoes`, `steel_food_cans`, `styrofoam_cups`, `styrofoam_food_containers`, `tea_bags`.

Mapping 30 fine class → 3 superclass (`Kitchen waste`, `Recyclable waste`, `Other waste`)
didefinisikan eksplisit di `CLASS_TO_MAJOR` pada Bagian 1 notebook. Mapping ini hanya
memengaruhi laporan superclass, bukan model 30 kelas. Kalau interpretasi kelas berbeda
(misalnya `paper_cups` recyclable atau tidak), edit dict tersebut sebelum training.

## Isi notebook

1. Setup dan konfigurasi (+ auto-install library yang belum ada)
2. Pencarian dataset (resolver multi-platform)
3. Exploratory data analysis (+ ringkasan keseimbangan kelas dan statistik gambar)
4. Split 8:1:1 stratified dengan dedupe MD5
5. Dataset, transform, dan loader
6. Model MobileNetV2
7. Training loop (pola `train_245_class.py`: deepcopy best weights, StepLR)
8. Training MobileNetV2
9. Kurva training
10. Evaluasi test: 30 kelas dan 3 superclass
11. Confusion matrix (insight tambahan)
12. Inference tiga gambar
13. CNN baseline (arsitektur `train_cnn.py` di PyTorch)
14. Analisis kesalahan testing (gambar salah prediksi, pasangan kelas tertukar)
15. Perbandingan model
16. Simpan ringkasan (`outputs/results/summary.json`)

## Output

Semua artefak ditulis ke `Reproduce Code/outputs/`:

```text
outputs/
├── figures/
│   ├── class_samples.png, class_distribution.png
│   ├── eda_class_balance.png, eda_image_stats.png
│   ├── mobilenetv2_curves.png, cnn_curves.png
│   ├── mobilenetv2_confusion_30.png, mobilenetv2_confusion_major.png
│   ├── cnn_confusion_30.png, cnn_confusion_major.png
│   ├── mobilenetv2_inference_examples.png
│   ├── mobilenetv2_misclassified_examples.png, cnn_misclassified_examples.png
│   └── model_comparison.png
├── models/
│   ├── mobilenetv2_trash30_full.pt, mobilenetv2_trash30_best.pt
│   └── cnn_baseline_trash30_full.pt, cnn_baseline_trash30_best.pt
├── results/
│   ├── mobilenetv2_history.csv, cnn_baseline_history.csv
│   ├── mobilenetv2_metrics.json, cnn_baseline_metrics.json
│   ├── mobilenetv2_classification_report.csv, mobilenetv2_superclass_report.csv
│   ├── mobilenetv2_per_class_accuracy.csv, mobilenetv2_predictions.csv
│   ├── cnn_baseline_classification_report.csv, cnn_baseline_superclass_report.csv
│   ├── cnn_baseline_per_class_accuracy.csv, cnn_baseline_predictions.csv
│   ├── mobilenetv2_misclassified.csv, cnn_baseline_misclassified.csv
│   ├── model_comparison.csv, model_insights.csv
│   └── summary.json
└── splits/
    ├── all.csv, train.csv, val.csv, test.csv
```

`*.pt` full model menyimpan seluruh model PyTorch, `*.pt` best checkpoint menyimpan
`state_dict` plus metadata (nama kelas, mapping superclass, ukuran gambar, konstanta
normalisasi) sehingga inferensi tidak bergantung pada daftar hard-coded di notebook.

`outputs/` dibuat saat notebook dijalankan dan tidak termasuk source delivery (checkpoint
sendiri sekitar 100 MB).

## Perbedaan dengan paper dan kode asli

| Aspek | Paper / `Source Code` | Notebook ini |
|---|---|---|
| Dataset | 245 fine class, sekitar 80.000 gambar | Kaggle 30 fine class, 15.000 file → 11.263 unik |
| Superclass | Kitchen, Recyclable, Hazardous, Other | Kitchen, Recyclable, Other (tidak ada kelas hazardous) |
| Nama kelas | `Major_minor`, misalnya `Kitchen waste_apple` | `snake_case`, misalnya `food_waste` |
| Label superclass | Prefix sebelum `_` | Mapping manual `CLASS_TO_MAJOR` |
| Split | `utils.py` menyalin file 8:1:1 tanpa seed | Manifest CSV stratified per kelas dan source group, seed 42 |
| Duplikasi | Tidak ada handling | MD5 dedupe sebelum split |
| Validation loader | `shuffle=True` | `shuffle=False` |
| Framework | PyTorch dan TensorFlow/Keras | PyTorch untuk MobileNetV2 dan CNN |
| Preprocessing CNN | `Rescaling(1/255)` + resize 224 | `Resize((224, 224))` + `ToTensor()` 0–1, tanpa augmentasi |
| Checkpoint CNN | Model epoch terakhir | Checkpoint validation terbaik |
| Penyimpanan model | Full model `.pt`, `.pth`, `.pkl` | Full model `.pt` dan checkpoint `state_dict` + metadata |
| Evaluasi | Accuracy dan loss di paper; `classification_report` 4 superclass di GitHub | Accuracy, loss, `classification_report` 3 superclass; metrik lain sebagai insight tambahan |
| Aplikasi | Flask backend (`main.py`) dan mobile app | Tidak diimplementasikan |

Paper melaporkan **82.92%** untuk MobileNetV2 dan **67.5%** untuk CNN pada dataset 245 kelas
mereka. Angka itu tidak bisa direplikasi persis tanpa dataset asli. Yang dibandingkan di
notebook adalah selisih antara kedua model pada split data yang sama dengan preprocessing
yang mengikuti paper/GitHub.

Paper juga tidak konsisten soal epoch MobileNetV2: bagian Methods dan `train_245_class.py`
memakai 50 epoch, caption Fig 5 menyebut 30. Notebook memakai 50 epoch.

## Catatan

* Notebook menyimpan ringkasan machine-readable di `outputs/results/summary.json`
  (info dataset, hyperparameter, metrik kedua model, versi library).
* Split bersifat deterministik (`SEED = 42`) dan disimpan sebagai CSV. Menjalankan ulang cell
  training tanpa menghapus `outputs/` memakai manifest yang sama.
* Deterministik GPU tidak bit-exact, selisih kecil antar-run normal.
* Smoke test cepat (lihat `Reproduce Code/README.md`): `TRASH_REPRO_QUICK=1` dan
  `TRASH_REPRO_PRETRAINED=0`. Hanya untuk memastikan notebook jalan, **jangan dipakai sebagai
  hasil yang dilaporkan**.
