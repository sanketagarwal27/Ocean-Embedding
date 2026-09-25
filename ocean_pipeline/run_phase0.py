"""
Phase 0 - Master run script.

Executes the full Phase 0 pipeline in order:
    1. Inspect raw files (catalogue what we have)
    2. Build ocean mask + coastal confidence mask + visualize
    3. Discover available dates in each file, test ingesters
    4. Build sample daily tensors using available file dates
    5. ARGO co-location self-test (synthetic profile)

NOTE: The current NC files are single-day snapshots for pipeline validation.
      Full multi-year downloads are required before Phase 1 training.

Run with:
    .\\venv\\Scripts\\python.exe ocean_pipeline\\run_phase0.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
import numpy as np
from datetime import date

print("\n" + "="*70)
print("  PHASE 0 -- DATA ENGINEERING PIPELINE")
print("="*70)


# -------------------------------------------------------------------------
# Step 1: Inspect raw files
# -------------------------------------------------------------------------
print("\n[STEP 1] Inspecting raw NetCDF files...")
from ocean_pipeline.ingestion.inspect_raw_files import main as inspect_main
inspect_main()


# -------------------------------------------------------------------------
# Step 2: Build masks
# -------------------------------------------------------------------------
print("\n[STEP 2] Building land/ocean and coastal confidence masks...")
from ocean_pipeline.preprocessing.build_masks import (
    build_ocean_mask, compute_distance_to_coast,
    build_confidence_mask, visualize_masks
)
from ocean_pipeline.config import DATA_MASKS, OUTPUTS_PLOTS, COAST_DISTANCE_KM, NLAT, NLON

ocean_mask = build_ocean_mask(source="ssh")
dist_km    = compute_distance_to_coast(ocean_mask)
confidence = build_confidence_mask(dist_km, threshold_km=COAST_DISTANCE_KM)

DATA_MASKS.mkdir(parents=True, exist_ok=True)
np.save(DATA_MASKS / "ocean_mask.npy",       ocean_mask)
np.save(DATA_MASKS / "dist_to_coast_km.npy", dist_km)
np.save(DATA_MASKS / "coast_confidence.npy", confidence)
print(f"  [OK] Masks saved to {DATA_MASKS}")
print(f"       Ocean cells: {ocean_mask.sum()} / {NLAT * NLON} "
      f"({100 * ocean_mask.mean():.1f}% ocean)")

print("  Generating mask visualizations...")
visualize_masks(ocean_mask, dist_km, confidence, OUTPUTS_PLOTS)


# -------------------------------------------------------------------------
# Step 3: Discover available dates + test ingesters
# -------------------------------------------------------------------------
print("\n[STEP 3] Discovering available dates in each file...")
from ocean_pipeline.ingestion.ingesters import (
    SSTIngester, SSHIngester, SSSIngester
)

ingesters_info = {}
for cls, name in [(SSTIngester, "SST"), (SSHIngester, "SSH"), (SSSIngester, "SSS")]:
    try:
        ing = cls()
        avail = ing.get_available_dates()
        ingesters_info[name] = {"ingester": ing, "dates": avail}
        print(f"  {name}: {len(avail)} date(s) in file -> {avail}")
    except Exception as e:
        ingesters_info[name] = {"ingester": None, "dates": []}
        print(f"  {name}: ERROR reading dates: {e}")

# Find a common test date (or best available per channel)
# Use whatever date is in each file
test_dates_per_channel = {}
for name, info in ingesters_info.items():
    if info["dates"]:
        test_dates_per_channel[name] = info["dates"][0]
        print(f"  Using {name} date: {test_dates_per_channel[name]}")

print("\n  Testing ingesters on their available dates...")
for cls, name in [(SSTIngester, "SST"), (SSHIngester, "SSH"), (SSSIngester, "SSS")]:
    if name not in test_dates_per_channel:
        print(f"  [SKIP] {name}: no available dates")
        continue
    d = test_dates_per_channel[name]
    dr = (d.strftime("%Y-%m-%d"), d.strftime("%Y-%m-%d"))
    try:
        da = cls().get_daily_field(date_range=dr)
        if len(da.time) > 0:
            print(f"  [OK] {name}: shape={da.shape}, "
                  f"mean={float(np.nanmean(da.values)):.3f}, "
                  f"NaN%={100*np.isnan(da.values).mean():.1f}%")
        else:
            print(f"  [WARN] {name}: 0 time steps returned")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")

# Test SSH geostrophic currents (OSCAR fallback)
print("\n  [SSH fallback] Geostrophic current components (OSCAR substitute)...")
try:
    ssh_ing = SSHIngester()
    ssh_avail = ssh_ing.get_available_dates()
    if ssh_avail:
        d_str = ssh_avail[0].strftime("%Y-%m-%d")
        u_geo, v_geo = ssh_ing.get_geostrophic_currents(date_range=(d_str, d_str))
        if u_geo is not None:
            print(f"  [OK] ugos shape={u_geo.shape}, "
                  f"mean={float(np.nanmean(u_geo.values)):.4f} m/s")
            print(f"  [OK] vgos shape={v_geo.shape}, "
                  f"mean={float(np.nanmean(v_geo.values)):.4f} m/s")
            print(f"  Note: geostrophic currents used as proxy until OSCAR downloaded")
except Exception as e:
    print(f"  [FAIL] SSH geostrophic: {e}")


# -------------------------------------------------------------------------
# Step 4: Build sample daily tensors using ACTUAL available file dates
# -------------------------------------------------------------------------
print("\n[STEP 4] Building daily input tensors using available file dates...")
from ocean_pipeline.preprocessing.build_tensors import (
    build_daily_tensor, visualize_sample_day
)
from ocean_pipeline.config import NCHAN, NDEPTH, INPUT_CHANNELS

# Collect all available dates across all files
all_available = set()
for name, info in ingesters_info.items():
    all_available.update(info["dates"])

if not all_available:
    print("  [WARN] No dates found in any file. Skipping tensor building.")
    print("  Download full time series to proceed.")
else:
    # Use available dates (may differ per channel -- that's documented)
    sample_dates = sorted(all_available)[:3]  # up to 3 dates
    print(f"  Sample dates to process: {sample_dates}")
    print(f"  Note: Channels with no data for a given date will be NaN (expected).")

    first_done = False
    for d in sample_dates:
        print(f"\n  Processing {d}...")
        try:
            inp, tgt, status = build_daily_tensor(d, ocean_mask=ocean_mask, save=True)

            assert inp.shape == (NLAT, NLON, NCHAN), f"Bad input shape: {inp.shape}"
            assert tgt.shape == (NLAT, NLON, NDEPTH), f"Bad target shape: {tgt.shape}"

            ocean_ok = inp[ocean_mask, :]
            print(f"  {'Channel':15s}  {'Status':12s}  {'NaN%':8s}  {'Mean':>10s}")
            print(f"  {'-'*50}")
            for i, ch in enumerate(INPUT_CHANNELS):
                col = ocean_ok[:, i]
                nan_pct = 100 * np.isnan(col).mean()
                val_mean = float(np.nanmean(col)) if not np.all(np.isnan(col)) else float('nan')
                stat = status.get(ch, "?")
                print(f"  {ch:15s}  {stat:12s}  {nan_pct:6.1f}%   {val_mean:10.3f}")

            print(f"\n  [OK] input={inp.shape}  target={tgt.shape}")

            if not first_done:
                print("  Generating diagnostic plots...")
                visualize_sample_day(inp, tgt, d, ocean_mask=ocean_mask)
                first_done = True

        except Exception as e:
            import traceback
            print(f"  [FAIL] {d}: {e}")
            traceback.print_exc()


# -------------------------------------------------------------------------
# Step 5: ARGO co-location self-test
# -------------------------------------------------------------------------
print("\n[STEP 5] ARGO co-location self-test (synthetic profile)...")
from ocean_pipeline.ingestion.argo_colocation import colocate_profile

lat, lon = 15.1, 75.3   # open Arabian Sea
d_argo = date(2023, 6, 1)
depths = np.array([0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 700], dtype=np.float32)
temps  = np.array([29, 28.5, 28, 27, 25, 22, 18, 15, 10, 8, 6, 4, 2], dtype=np.float32)
salts  = np.array([35.0] * 13, dtype=np.float32)

result = colocate_profile(lat, lon, d_argo, depths, temps, salts)
if result:
    print(f"  [OK] Co-located to grid cell "
          f"({result['grid_lat_idx']}, {result['grid_lon_idx']})")
    print(f"       Valid depth levels: {result['n_valid_depths']}/15")
    print(f"       T at standard depths: {result['temp_15lev']}")
else:
    print("  [FAIL] Co-location returned None")


# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
from ocean_pipeline.config import (
    LATS, LONS, GRID_RES,
    TRAIN_YEARS, VAL_YEARS, TEST_YEARS
)

print("\n" + "="*70)
print("  PHASE 0 ACCEPTANCE CRITERIA SUMMARY")
print("="*70)

print(f"""
Grid:          {NLAT}x{NLON} @ {GRID_RES}deg,
               lat=[{LATS[0]}-{LATS[-1]}N], lon=[{LONS[0]}-{LONS[-1]}E]
Input tensor:  [H={NLAT}, W={NLON}, C={NCHAN}]  channels: {INPUT_CHANNELS}
Target tensor: [H={NLAT}, W={NLON}, D={NDEPTH}] depth levels: 0m to 1000m
Ocean mask:    {ocean_mask.sum()} ocean cells / {NLAT*NLON} total ({100*ocean_mask.mean():.1f}% ocean)
Coastal mask:  {COAST_DISTANCE_KM}km threshold, sigmoid transition
Ingesters:     SST/SSH/SSS present (single-day snapshots); OSCAR/CCMP/GLORYS stubbed
SSH fallback:  ugos/vgos geostrophic currents available as OSCAR proxy
Plots:         ocean_mask.png, coastal_confidence.png, sample_day_inputs_*.png
Train split:   {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]} | Val: {VAL_YEARS} | Test: {TEST_YEARS}
ARGO:          held-out; never used in training

BLOCKERS before Phase 1 (must resolve):
  - Download full time series (2015-2023) for SST, SSH, SSS from CMEMS
  - Download OSCAR L4 currents (U,V) from PODAAC
  - Download CCMP V3.1 winds (U,V) from PODAAC
  - Download GLORYS T,S (training target) from CMEMS
  - Download gridded ARGO from INCOIS LAS (validation set -- held out)

Regridding deviation:
  xesmf (ESMF) not used -- compiled Fortran not available on Windows.
  pyresample (scipy/KD-tree) used instead; documented throughout codebase.
""")

print("  Phase 0 pipeline infrastructure: COMPLETE")
print("  Awaiting full data downloads before Phase 1 training begins.")
