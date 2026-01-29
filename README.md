# SCCP Regression

This repository contains a regression-version implementation of
Shrinkage-Clustered Conformal Prediction (SCCP),
with a focus on uncertainty quantification for heavy-tailed and imbalanced
continuous outcomes.

The primary application in this project is **photometric redshift estimation**
using galaxy photometry from the Hyper Suprime-Cam (HSC) survey,
but the framework is designed to be applicable to general regression problems.

---

## Project Overview

- **Task**: Regression with distribution-free uncertainty quantification
- **Method**: Split conformal prediction and shrinkage-clustered conformal prediction (SCCP)
- **Data**: HSC PDR2 photometry with spectroscopic redshifts
- **Setting**:
  - Raw numeric covariates (no images, no embeddings)
  - Strong imbalance and heavy-tailed response distribution
  - Emphasis on stability in sparse regions of the covariate space

---
## Data Splitting

The HSC dataset is randomly split into:
- 70% training
- 10% validation
- 10% conformal calibration
- 10% testing

The calibration set is held out and used exclusively for conformal inference.

---
## Run Simulation
```bash
python scripts/run_hsc_cp_rf.py \
  --M 100 \
  --gamma 0.8 \
  --Kbin 1000 \
  --bin_mode tail \
  --tau 10
```
#### Options
-  `--M`
- `--gamma`
- `--Kbin`
- `--bin_mode`
- `--tau`
