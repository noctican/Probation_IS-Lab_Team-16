# Reproduce — Application of MobileNetV2 to Waste Classification

Reproduction of: `../Paper/journal.pone.0282336.pdf`

> Yong, L., Ma, L., Sun, D., & Du, L. (2023). **Application of MobileNetV2 to waste
> classification.** *PLOS ONE*, 18(3), e0282336.
> https://doi.org/10.1371/journal.pone.0282336

The original code is in `../Source Code/`. This folder contains a rewritten, self-contained
notebook that runs on **Kaggle**, **Google Colab**, and **locally**.

**Summary:** the notebook uses the Kaggle 30-class dataset (15,000 files across the `default`
and `real_world` source groups), removes exact duplicates by MD5 (11,263 unique images remain),
splits the data 8:1:1 in a stratified way (9,006 train / 1,124 validation / 1,133 test), trains
MobileNetV2 and a CNN baseline, and evaluates 30-class accuracy plus 3 superclasses (Kitchen,
Recyclable, Other). The paper's original 245-class dataset is not available, and this dataset has
no hazardous class.

## Files

| File | Purpose |
|---|---|
| `Reproduce_MobileNetV2_Waste_Classification.ipynb` | Main notebook: data discovery, EDA, 8:1:1 split, MobileNetV2 training/evaluation, CNN baseline, error analysis, comparison, inference |
| `requirements.txt` | Python packages for a local run |
| `outputs/` | Generated artifacts (created by the notebook, not part of the source delivery) |

The mobile application of the paper and of the original repository (Flask backend `main.py` and
the WeChat applet frontend) is **not implemented**. The reproduction focuses on the computer
vision pipeline: dataset preparation, training, evaluation, and single-image inference.

## Dataset

**Kaggle:** [Recyclable and Household Waste Classification](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification)
(Alistair King, 2024).

The dataset has 30 balanced classes and 15,000 PNG images:

```
recyclable-and-household-waste-classification/
└── images/
    └── images/
        ├── aerosol_cans/
        │   ├── default/     (250 images)
        │   └── real_world/  (250 images)
        ├── aluminum_food_cans/
        ...
        └── tea_bags/
```

Classes: `aerosol_cans`, `aluminum_food_cans`, `aluminum_soda_cans`, `cardboard_boxes`,
`cardboard_packaging`, `clothing`, `coffee_grounds`, `disposable_plastic_cutlery`, `eggshells`,
`food_waste`, `glass_beverage_bottles`, `glass_cosmetic_containers`, `glass_food_jars`, `magazines`,
`newspaper`, `office_paper`, `paper_cups`, `plastic_cup_lids`, `plastic_detergent_bottles`,
`plastic_food_containers`, `plastic_shopping_bags`, `plastic_soda_bottles`, `plastic_straws`,
`plastic_trash_bags`, `plastic_water_bottles`, `shoes`, `steel_food_cans`, `styrofoam_cups`,
`styrofoam_food_containers`, `tea_bags`.

| Stage | Count |
|---|---:|
| Files found | 15,000 |
| Exact duplicates removed (MD5) | 3,737 |
| Unique images | 11,263 |
| Train / Validation / Test | 9,006 / 1,124 / 1,133 |

Both source groups are kept (`default` 7,019 images, `real_world` 4,244 images). Duplicates are
removed **before** the split and the result is re-checked, so no MD5 appears in two different
splits.

## How the notebook finds the data

| Environment | Behaviour |
|---|---|
| **Kaggle** | Detects the attached dataset under `/kaggle/input/recyclable-and-household-waste-classification/...`. **No download is attempted.** Attach it with **Add Data → Datasets** before running. |
| **Google Colab** | Tries `kagglehub.dataset_download("alistairking/recyclable-and-household-waste-classification")`, then the `kaggle` CLI, then the direct Kaggle API URL. The dataset is public, so a token is normally not needed. If Kaggle requests authentication, add a Colab secret named `KAGGLE_API_TOKEN`. |
| **Local** | Looks for `Datasets/images/images/` relative to the notebook/repository, then for `TRASH_DATA_DIR`. |

For a local run from this folder, the default path is:

```
../Datasets/images/images/
```

You can also set an explicit path:

```bash
export TRASH_DATA_DIR="/absolute/path/to/Datasets/images/images"
jupyter notebook
```

## Running

### Kaggle

1. Create a notebook and attach the dataset (`Add Data` → search the dataset name).
2. Upload/import `Reproduce_MobileNetV2_Waste_Classification.ipynb`.
3. Enable a GPU accelerator (optional but much faster).
4. Run all cells.

### Google Colab

1. Upload the notebook.
2. Select **Runtime → Change runtime type → GPU** (optional).
3. Run all cells. The dataset downloads automatically.
4. If the download fails, add a `KAGGLE_API_TOKEN` secret or upload the dataset manually and set
   `TRASH_DATA_DIR` in Section 1.

### Local

```bash
python -m pip install -r requirements.txt
jupyter notebook Reproduce_MobileNetV2_Waste_Classification.ipynb
```

The local `Datasets/` folder is used automatically.

## Notebook structure

| Section | Content |
|---|---|
| 1 | Setup and configuration (paths, hyperparameters, runtime detection) |
| 2 | Dataset search (multi-platform resolver) |
| 3 | Exploratory data analysis + class-balance and image-statistics summary |
| 4 | Stratified 8:1:1 split with MD5 deduplication |
| 5 | Dataset, transforms, and dataloaders |
| 6 | MobileNetV2 model (ImageNet transfer learning) |
| 7 | Training loop (`train_245_class.py` pattern: `copy.deepcopy` best weights, StepLR) |
| 8 | MobileNetV2 training |
| 9 | Training curves |
| 10 | Test evaluation: 30 classes and 3 superclasses |
| 11 | Confusion matrices (additional insight) |
| 12 | Inference on three images |
| 13 | CNN baseline (`train_cnn.py` architecture reimplemented in PyTorch) |
| 14 | Test error analysis (misclassified images, most confused class pairs) |
| 15 | Model comparison |
| 16 | Machine-readable summary (`outputs/results/summary.json`) |

## Output

All artifacts are written under `outputs/` next to the notebook:

```
outputs/
├── figures/
│   ├── class_distribution.png
│   ├── class_samples.png
│   ├── eda_class_balance.png
│   ├── eda_image_stats.png
│   ├── mobilenetv2_curves.png
│   ├── mobilenetv2_confusion_30.png
│   ├── mobilenetv2_confusion_major.png
│   ├── cnn_curves.png
│   ├── cnn_confusion_30.png
│   ├── cnn_confusion_major.png
│   ├── mobilenetv2_inference_examples.png
│   ├── mobilenetv2_misclassified_examples.png
│   ├── cnn_misclassified_examples.png
│   └── model_comparison.png
├── models/
│   ├── mobilenetv2_trash30_full.pt
│   ├── mobilenetv2_trash30_best.pt
│   ├── cnn_baseline_trash30_full.pt
│   └── cnn_baseline_trash30_best.pt
├── results/
│   ├── mobilenetv2_history.csv
│   ├── mobilenetv2_metrics.json
│   ├── mobilenetv2_classification_report.csv
│   ├── mobilenetv2_superclass_report.csv
│   ├── mobilenetv2_per_class_accuracy.csv
│   ├── mobilenetv2_predictions.csv
│   ├── cnn_baseline_history.csv
│   ├── cnn_baseline_metrics.json
│   ├── cnn_baseline_classification_report.csv
│   ├── cnn_baseline_superclass_report.csv
│   ├── cnn_baseline_per_class_accuracy.csv
│   ├── cnn_baseline_predictions.csv
│   ├── mobilenetv2_misclassified.csv
│   ├── cnn_baseline_misclassified.csv
│   ├── model_comparison.csv
│   ├── model_insights.csv
│   └── summary.json
└── splits/
    ├── all.csv
    ├── train.csv
    ├── val.csv
    └── test.csv
```

`outputs/` is generated when the notebook runs; it is not part of the source delivery (the
checkpoints alone are ~100 MB). The `*_full.pt` files store the complete PyTorch model; the
`*_best.pt` checkpoints store the best-validation `state_dict` together with metadata (class
names, superclass mapping, image size, normalization constants) so that inference does not
depend on the notebook's hard-coded lists.

The notebook also provides `predict_single_image(...)`, a single-image inference helper used in
Section 12 to classify three test images (true label, predicted class, top-5 probabilities). No
web service is generated: the Flask backend of the original repository (`../Source Code/main.py`)
and the WeChat applet frontend are not reproduced here.

## Evaluation protocol

The primary evaluation mirrors `test_model.py` from the original repository:

1. 30-class test accuracy,
2. cross-entropy loss on the test set,
3. `classification_report` for the 3 superclasses, where the superclass of a sample is derived
   from its top-1 fine class through `CLASS_TO_MAJOR` (the GitHub version derives it from the
   `Major_minor` prefix).

Everything else — top-3 accuracy, macro F1, 3-class accuracy from aggregated superclass
probabilities, confusion matrices, error analysis — is reported separately as *additional
insight* (`model_insights.csv`) and does not change the primary evaluation.

## Reproduction choices and differences from the paper

| Item | Paper | This notebook |
|---|---|---|
| Dataset | Original 245-class household-waste dataset, >80k images | Public Kaggle 30-class dataset, 15,000 files → 11,263 unique |
| Superclasses | Recyclable, kitchen, hazardous, other | Kitchen, recyclable, other (the Kaggle dataset has **no hazardous class**) |
| Class names | `Major_minor`, e.g. `Kitchen waste_apple` | `snake_case`, e.g. `food_waste` |
| Superclass label | Prefix before `_` | Explicit `CLASS_TO_MAJOR` mapping |
| Split | 8:1:1 stratified | 8:1:1 stratified by class **and** `default`/`real_world` source group, `SEED = 42` |
| Duplicates | Not handled | MD5 dedupe before the split, so no image appears in two splits |
| Validation loader | `shuffle=True` | `shuffle=False` |
| MobileNetV2 | Transfer learning, 50 epochs, SGD 0.001, momentum 0.9, StepLR(7, 0.1) | Same |
| CNN baseline | TensorFlow/Keras, Adam, 30 epochs | Same architecture in PyTorch, Adam, 30 epochs, batch 16 |
| CNN preprocessing | `Rescaling(1/255)` + resize to 224 | `Resize((224, 224))` + `ToTensor()` (0–1), no augmentation, no ImageNet normalization |
| Framework | Mixed PyTorch and TensorFlow | PyTorch throughout, so both models share one data pipeline |
| Checkpoint CNN | Model of the last epoch | Best-validation checkpoint |
| Saved models | Full model `.pt`, `.pth`, `.pkl` | Full model `.pt` **and** `state_dict` checkpoints with metadata |
| Evaluation | Accuracy and loss in the paper; `classification_report` on 4 superclasses in GitHub | Accuracy, loss, `classification_report` on 3 superclasses; other metrics as additional insight |
| Application | Flask backend and mobile app | Not implemented |

The paper reports **82.92%** for MobileNetV2 and **67.5%** for CNN. Those numbers cannot be
reproduced exactly without the original 245-class data. This notebook reports its own numbers on
the Kaggle dataset, plus the same comparison (same split and preprocessing) between the two
models.

The paper is also inconsistent about the MobileNetV2 epoch count: the Methods section and
`train_245_class.py` use 50 epochs, while the caption of Fig. 5 mentions 30. The notebook uses
50 epochs.

The 30-class → 3-superclass mapping is defined explicitly in Section 2 of the notebook. If your
interpretation of a class differs (for example, whether `paper_cups` is recyclable), edit
`CLASS_TO_MAJOR` before training. The mapping does not affect the 30-class model; it only affects
the superclass report.

## Quick smoke test

To verify the notebook without waiting for full training, set:

```bash
export TRASH_REPRO_QUICK=1
export TRASH_REPRO_PRETRAINED=0
```

This trains one epoch on 20 images/class and disables ImageNet weight downloads. Do not use the
resulting metrics for reporting.

## Notes

* Training uses ImageNet normalization and `RandomResizedCrop(224)` + horizontal flip, matching
  `train_245_class.py`.
* The checkpoint stores class names, the class-to-superclass mapping, image size and normalization
  constants, so inference is independent of the notebook's hard-coded lists.
* The split is deterministic (`SEED = 42`) and saved as CSV. Re-running the training cells without
  deleting `outputs/` reuses the same manifests.
* `outputs/results/summary.json` stores the dataset description, hyperparameters, metrics of both
  models, artifact paths, and library versions in a machine-readable form.
* GPU determinism is not bit-exact; small run-to-run differences are normal.
