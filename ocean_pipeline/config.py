"""
Central configuration for the Ocean Embedding pipeline.
All constants, domain specs, file paths, and hyperparameters live here.
"""
import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_MASKS     = ROOT / "data" / "masks"
DATA_ARGO      = ROOT / "data" / "argo_validation"
LOGS_DIR       = ROOT / "logs"
OUTPUTS_PLOTS  = ROOT / "outputs" / "plots"
OUTPUTS_CKPT   = ROOT / "outputs" / "checkpoints"

# Create all directories on import
for d in [DATA_RAW, DATA_PROCESSED, DATA_MASKS, DATA_ARGO,
          LOGS_DIR / "tensorboard", OUTPUTS_PLOTS, OUTPUTS_CKPT]:
    d.mkdir(parents=True, exist_ok=True)

# ── Domain specification ───────────────────────────────────────────────────────
# North Indian Ocean: Arabian Sea + Bay of Bengal
LAT_MIN =  5.0   # °N
LAT_MAX = 30.0   # °N
LON_MIN = 45.0   # °E
LON_MAX = 105.0  # °E
GRID_RES = 0.25  # degrees

# Derived grid size
import numpy as np
LATS = np.arange(LAT_MIN, LAT_MAX + GRID_RES, GRID_RES)
LONS = np.arange(LON_MIN, LON_MAX + GRID_RES, GRID_RES)
NLAT = len(LATS)   # 101
NLON = len(LONS)   # 241

# ── Depth levels (target) ──────────────────────────────────────────────────────
DEPTH_LEVELS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
NDEPTH = len(DEPTH_LEVELS)   # 15

# ── Input channels ────────────────────────────────────────────────────────────
# Order must be consistent everywhere — changes here break checkpoints!
INPUT_CHANNELS = ["sst", "sss", "ssh", "u_current", "v_current", "u_wind", "v_wind"]
NCHAN = len(INPUT_CHANNELS)   # 7

# ── Coastal confidence mask ───────────────────────────────────────────────────
COAST_DISTANCE_KM = 50.0   # cells within this distance flagged as low-confidence

# ── Normalization ─────────────────────────────────────────────────────────────
# Method: z-score per channel across training period
# Stats will be computed and saved as masks/norm_stats.npz
NORM_METHOD = "zscore"   # alternative: "minmax"

# ── Time splits ───────────────────────────────────────────────────────────────
# Adjust after confirming actual data availability
TRAIN_YEARS = list(range(2015, 2022))   # 2015–2021
VAL_YEARS   = [2022]
TEST_YEARS  = [2023]

# ── Training hparams (placeholder — overridden per phase) ─────────────────────
SEED = 42
BATCH_SIZE = 4
LR = 1e-4
NUM_EPOCHS = 50
TEMPORAL_WINDOW = 7   # days stacked for Phase 1b+

# ── File patterns (raw downloads) ─────────────────────────────────────────────
# These are the filenames already present in the workspace root.
# Ingest scripts will copy/symlink them to data/raw/.
EXISTING_NC_FILES = {
    "sst": ROOT / "METOFFICE-GLO-SST-L4-REP-OBS-SST_1790354343090.nc",
    "ssh": ROOT / "c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D_1790354504516.nc",
    "sss": ROOT / "cmems_obs-mob_glo_phy-sss_my_multi_P1D_1790354487475.nc",
}

print(f"[config] Grid: {NLAT}×{NLON} ({GRID_RES}°), {NDEPTH} depth levels, {NCHAN} channels")
