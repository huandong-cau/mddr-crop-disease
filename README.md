# MDDR: Metacognitive Doubt-Driven Reasoning for Crop Disease Diagnosis

Official code for the manuscript **"Metacognitive Doubt-Driven Reasoning for Trustworthy Crop Disease Diagnosis: Triadic Uncertainty Decomposition with Agronomic Grounding"** (Huandong Wang, Qingdao Agricultural University; under review).

MDDR decomposes a crop-disease classifier's uncertainty into three actionable doubts — aleatoric, epistemic and ontological — and couples them to an agronomically-grounded replanning (AGCR) policy that escalates with a top-3 differential diagnosis instead of acting when doubt fires.

## Key results (mean ± SD over 10 seeds)

| Metric | Best baseline | MDDR |
|---|---|---|
| OOD-detection AUROC (phenotypic shift) | 0.748 (EDL) | **0.790** |
| Agronomic validity (deployment set) | 0.751 | **1.000** |
| Incorrect autonomous actions | 0.178 | **0.091 (−48.9%)** |
| Escalation precision | — | 0.734 |
| Label-noise audit precision@100 | 0.076 (random) | **0.211 (2.8×)** |

Headline comparisons: Wilcoxon W=0, p=0.002 (n=10), Cliff's δ=1.00.

## Data

Maize subset of the PlantVillage-derived augmented collection (gray leaf spot, northern corn leaf blight, common rust, healthy; 4,354 images; public release). The pipeline expects the class folders under:

```
Plant_leave_diseases_dataset_with_augmentation/
  Corn___Cercospora_leaf_spot Gray_leaf_spot/
  Corn___Northern_Leaf_Blight/
  Corn___Common_rust/
  Corn___healthy/
```

Paths are configured in `md_pipeline/common.py`.

## Pipeline

```bash
python md_pipeline/01_prep_data.py        # near-dup-aware split, OOD shifts, deployment metadata
# place resnet50-0676ba61.pth at weights/resnet50-0676ba61.pth
#   (https://download.pytorch.org/models/resnet50-0676ba61.pth, 97.8 MB)
python md_pipeline/02_extract_features.py # frozen ResNet-50 features for all subsets
python md_pipeline/03_train_models.py     # 4 baselines x 5 seeds + label-noise models
python md_pipeline/04_mddr.py             # triadic doubt, AGCR policy, AVD audit
python md_pipeline/05_evaluate.py         # metrics, statistics, tables, figures
```

Environment: Python 3.13, PyTorch (CPU is sufficient), numpy, scipy, scikit-learn, Pillow, matplotlib. All experiments run on a single 18-core consumer workstation; the frozen-feature design keeps training tractable without a GPU.

## Reproducibility

- Ten seeds (1–10) throughout; split seed 42; near-duplicate grouping by 64-bit perceptual hash (Hamming ≤ 4).
- Deterministic stressors: phenotypic shift (saturation ×0.50, contrast ×1.45, brightness ×1.30, Gaussian blur σ=1.5), ontological violations (25% staged metadata conflicts against the rule table in `common.py`), label noise (20% GLS↔NCLB flip).
- Thresholds: τe selected on the validation shift set by Youden's J; τa = 0.95.
- All intermediate predictions and per-seed metrics are cached under `md_pipeline/cache/`; final tables and 300-dpi figures under `results/`.

## License

MIT. The image collection remains under its original license (public PlantVillage-derived release).

## Contact

Huandong Wang, College of Science and Information Science, Qingdao Agricultural University — 2899643179@qq.com
