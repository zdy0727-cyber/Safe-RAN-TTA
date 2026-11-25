# Safe-RAN-TTA

Code for the paper **"Calibrated test-time adaptation with risk-aware decision for cross-dataset
winter road-surface recognition"** (submitted to *Optical Engineering*).

The repository contains the training, test-time adaptation (TTA), evaluation and plotting scripts
used to produce the results in the paper. Data are **not** included; the four public datasets are
available from their original sources:

- RoadSaW-12 — https://roadsaw.viscoda.com
- RoadSC — https://roadsc.viscoda.com
- RSCD (Tsinghua) — https://data.mendeley.com/datasets/w86hvkrzc5/3
- Xinjiang Tianshan imagery — see Remote Sensing 16(19):3727 (2024)

## Environment

- Python 3.10, PyTorch 2.5.1 + CUDA 12.1, torchvision, timm, numpy, pandas, matplotlib
- One or two NVIDIA GPUs (24 GB is sufficient for ResNet-50 at 224x224)

```
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm numpy pandas matplotlib
```

## Repository layout

| File | Purpose |
|---|---|
| `build_manifests.py` | Build Track-1 / Track-2 manifest CSVs from the public datasets |
| `build_multitask_manifests.py` | Mine auxiliary masks (wetness, snow cover, roughness, material) |
| `train_cls.py` | Single-task ResNet-50 training / evaluation baseline |
| `train_mt.py` | Multi-task source model training (`--head-mode full`, `--tasks`) |
| `tta_mt.py` | Test-time adaptation methods: `none`, `bn`, `tent`, `tent_risk`, `sar`, `eata`, `deyo`, `cotta`, `ran_tta`, `ran_safe`, plus calibrated adopt/abstain (`adapt_abstain`) and risk-aware decision (`--risk --risk-lam`) |
| `evaluate_tta.py` | Evaluate adapted/source models on target domains |
| `eval_in_domain.py` | In-domain metrics with present-label macro-F1 |
| `openset_scores.py` | Open-set scores (MSP, energy, Mahalanobis, prototype, kNN, dual score) |
| `collect_results.py` | Collect `*_summary.json` / `test_results.json` into tables |
| `plot_oe_figs.py` | Regenerate the result figures (single-column, 8 pt, 600 dpi) |

## Quick start

```bash
# 1) manifests
python build_manifests.py --root /path/to/data --out manifests/
python build_multitask_manifests.py --root /path/to/data --out manifests/multitask_all.csv

# 2) source model (Track-1, 15 classes, multi-task)
python train_mt.py --track track1 --num-classes 15 --head-mode full \
  --manifest manifests/multitask_all.csv --tasks wet,snow,rough,mat,water \
  --epochs 25 --bs 128 --lr 1e-4 --out runs/track1_src

# 3) test-time adaptation with calibrated adopt/abstain gate
python tta_mt.py --ckpt runs/track1_src/ckpt_best.pth --datasets ctd2 \
  --methods ran_safe --calib-split calib --calib-per-class 60 --calib-seed 0 \
  --out runs/ctd2_ran_safe

# 4) risk-aware decision (lambda = 0.2)
python tta_mt.py --ckpt runs/track1_src/ckpt_best.pth --datasets ctd2 \
  --methods ran_safe --risk --risk-lam 0.2 --out runs/ctd2_risk

# 5) figures
python plot_oe_figs.py
```

`tta_mt.py --help` lists all arguments (`--calib-split`, `--calib-per-class`, `--calib-seed`,
`--source-datasets`, `--unknown-classes`, `--risk`, `--risk-lam`, ...).

## Notes

- The calibrated gate uses a labeled calibration split that is disjoint from the test stream;
  with `--methods ran_safe` the candidate adaptation is adopted only when accuracy improves by more
  than 1% and macro-F1 does not decrease, otherwise the full model state (including BN buffers) is
  restored.
- Risk weights used in the paper: dry 1, wet 3, water 7, slush 8, snow 9, ice 10.
- The code is provided for research reproducibility. If you use it, please cite the paper.

## Citation

```
D. Zhou, Z. Liu, R. Song, K. Li, S. Chen, Z. Wang, Y. Li, and X. Qi,
"Calibrated test-time adaptation with risk-aware decision for cross-dataset
winter road-surface recognition," Optical Engineering (submitted, 2026).
```
