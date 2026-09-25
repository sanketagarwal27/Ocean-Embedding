# Ocean Embedding — North Indian Ocean Subsurface Temperature Reconstruction

Deep learning system reconstructing 3D subsurface ocean temperature from daily satellite surface observations.

**Domain**: North Indian Ocean, 5–30°N, 45–105°E | **Resolution**: 0.25° × 0.25° daily | **Depth levels**: 0–1000 m (15 levels)

---

## Architecture

```
Surface Inputs [H, W, 7]
      │
      ▼
Swin-Transformer Encoder   (Phase 2)
      │  per-cell latent vectors
      ▼
Advective Graph Neural Network   (Phase 3)
      │  current-gated message passing
      ▼
Conditional INR Decoder   (Phase 4)
      │  query at arbitrary depth z
      ▼
T(z), S(z)   [H, W, 15] at standard depths
      │
Physics Losses   (Phase 5)
L_MSE + L_density + L_wind (SoftAdapt weighted)
```

---

## Installation

### 1. Create virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install core dependencies
```powershell
pip install -r requirements.txt
```

### 3. Install PyTorch Geometric (after torch)
```powershell
pip install torch-geometric==2.6.1
```

> **⚠️ xesmf deviation**: The spec calls for `xesmf` (ESMF-based regridding), but ESMF requires a compiled Fortran library not pip-installable on Windows. `pyresample` (scipy/KD-tree based) is used instead, producing equivalent bilinear/nearest-neighbour results. This is documented throughout the codebase.

---

## Data Downloads Required

| Variable | Source | Status | URL |
|---|---|---|---|
| SST (OSTIA) | CMEMS | ✅ Present | https://doi.org/10.48670/moi-00168 |
| SSH (DUACS) | CMEMS | ✅ Present | https://doi.org/10.48670/moi-00145 |
| SSS (SMAP/SMOS) | CMEMS | ✅ Present | https://doi.org/10.48670/moi-00051 |
| Currents U,V (OSCAR) | PODAAC | ❌ Missing | https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_FINAL_V2.0 |
| Winds U,V (CCMP) | PODAAC | ❌ Missing | https://podaac.jpl.nasa.gov/dataset/CCMP_WINDS_10M6HR_L4_V3.1 |
| Target T,S (GLORYS) | CMEMS | ❌ Missing | https://doi.org/10.48670/moi-00021 |
| ARGO gridded | INCOIS LAS | ❌ Missing | https://las.incois.gov.in |

**ARGO data must never be used in training** — it is the independent validation set.

### Download OSCAR currents (PODAAC)
```bash
# Requires NASA Earthdata account
# OSCAR L4 OC FINAL V2.0 — 0.25°, daily
# See: https://opendap.earthdata.nasa.gov/
```

### Download GLORYS reanalysis (CMEMS)
```bash
# Requires CMEMS account (free registration at https://marine.copernicus.eu/)
# Product: GLOBAL_MULTIYEAR_PHY_001_030
# Variables: thetao (temperature), so (salinity)
# Domain: 5–30°N, 45–105°E
# Time: 2015-01-01 to 2023-12-31
copernicusmarine subset \
  --dataset-id cmems_mod_glo_phy_my_0.083deg_P1D-m \
  --variable thetao --variable so \
  --minimum-latitude 5 --maximum-latitude 30 \
  --minimum-longitude 45 --maximum-longitude 105 \
  --start-datetime 2015-01-01T00:00:00 \
  --end-datetime 2023-12-31T23:59:59 \
  --output-filename glorys_nio_2015_2023.nc
```

---

## Running Phase 0

```powershell
# Full Phase 0 pipeline (inspect → masks → ingest → tensors → plots)
.\venv\Scripts\python.exe ocean_pipeline\run_phase0.py

# Unit tests only
.\venv\Scripts\python.exe -m pytest ocean_pipeline\tests\test_preprocessing.py -v
```

**Expected outputs**:
- `outputs/plots/ocean_mask.png` — land/ocean mask
- `outputs/plots/coastal_confidence.png` — distance-to-coast + confidence masks
- `outputs/plots/sample_day_inputs_*.png` — 7-channel input diagnostic
- `data/masks/ocean_mask.npy`, `coast_confidence.npy`, `dist_to_coast_km.npy`
- `data/processed/sample_days/*.npz` — cached daily tensors

---

## Project Structure

```
ocean_pipeline/
├── config.py               — domain, paths, hyperparams
├── run_phase0.py           — master Phase 0 orchestrator
├── ingestion/
│   ├── ingesters.py        — per-dataset ingest classes
│   └── inspect_raw_files.py
├── preprocessing/
│   ├── regrid.py           — pyresample-based regridding
│   ├── build_masks.py      — ocean mask + coastal confidence
│   ├── build_tensors.py    — daily [H,W,C] tensor stacking
│   └── normalize.py        — z-score normalization
├── models/                 — (Phase 1+)
├── training/               — (Phase 1+)
├── evaluation/             — (Phase 1+)
├── diagnostics/            — (Phase 6)
└── tests/
    └── test_preprocessing.py

data/
├── raw/                    — symlinks/copies of raw NC files
├── processed/sample_days/  — cached .npz tensors
├── masks/                  — ocean_mask, coast_confidence, norm_stats
└── argo_validation/        — held-out ARGO (never train on this!)

outputs/
├── plots/                  — diagnostic PNGs
└── checkpoints/            — model checkpoints (Phase 1+)

logs/tensorboard/           — training logs (Phase 1+)
```

---

## Train/Val/Test Splits

| Split | Years | Purpose |
|---|---|---|
| Train | 2015–2021 | Model training |
| Val | 2022 | Hyperparameter tuning / early stopping |
| Test | 2023 | Final evaluation (used once) |
| ARGO | All years | Independent validation — **never touched during training** |

---

## Normalization

**Method**: z-score per channel (µ=0, σ=1), computed over ocean cells in the training period.

**Rationale**: preferred over min-max because ocean variable distributions are heavy-tailed, and z-score is robust to outliers from sparse-coverage satellite products (especially SSS and currents). Out-of-distribution test values do not saturate.

Stats saved to: `data/masks/norm_stats.npz`

---

## Phase Status

| Phase | Status | Acceptance Criteria |
|---|---|---|
| 0 — Data Engineering | 🔄 In Progress | ✅ Grid/mask/ingester/tensor pipeline built; ⚠️ OSCAR/CCMP/GLORYS/ARGO pending download |
| 1 — Baseline CNN+MLP | ⏳ Pending Phase 0 completion | — |
| 2 — Swin Encoder | ⏳ | — |
| 3 — Advective GNN | ⏳ | — |
| 4 — INR Decoder | ⏳ | — |
| 5 — Physics Losses | ⏳ | — |
| 6 — Validation | ⏳ | — |
| 7 — Demo | ⏳ | — |
