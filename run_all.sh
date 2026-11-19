#!/bin/bash
set -e

# Folder tempat semua notebook berada (relatif terhadap lokasi script dijalankan)
NB_DIR="Algorithm Exploration"

# Daftar notebook (hanya nama file, tanpa path)
NOTEBOOKS=(
  "EfficientFormerV2S0_ImageNetNorm_Waste_Classification.ipynb"
  "StarNetS1_ImageNetNorm_Waste_Classification.ipynb"
  "StarNetS2_ImageNetNorm_Waste_Classification.ipynb"
  "RepViTM09_ImageNetNorm_Waste_Classification.ipynb"
  "MobileNetV3Large_Torchvision_Waste_Classification.ipynb"
)

# Pastikan folder notebook ada
if [[ ! -d "$NB_DIR" ]]; then
  echo ">>> ERROR: Folder '$NB_DIR' tidak ditemukan. Jalankan script dari parent folder."
  exit 1
fi

for nb in "${NOTEBOOKS[@]}"; do
  in_path="$NB_DIR/$nb"
  out_path="$NB_DIR/Output_${nb}"

  if [[ ! -f "$in_path" ]]; then
    echo ">>> Dilewati (tidak ditemukan): $in_path"
    continue
  fi

  echo ""
  echo "============================================================"
  echo ">>> Mulai   : $in_path"
  echo ">>> Output  : $out_path"
  echo ">>> Waktu   : $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"

  # --cwd diarahkan ke folder notebook, agar path relatif di dalam notebook konsisten
  papermill "$in_path" "$out_path" \
    --progress-bar --log-output --cwd "$(pwd)/$NB_DIR"

  echo ">>> Selesai : $nb  ($(date '+%Y-%m-%d %H:%M:%S'))"
done

echo ""
echo ">>> Semua notebook selesai diproses."
