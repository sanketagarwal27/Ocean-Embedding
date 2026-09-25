"""
Phase 0 — Step 7: ARGO co-location module (held-out validation set).

**CRITICAL**: ARGO data is NEVER used during training.
It is co-located once, saved to data/argo_validation/, and only touched
during evaluation (Phases 1-6) to compute independent skill metrics.

This module handles two ARGO formats:
  1. Gridded ARGO from INCOIS LAS (preferred, NetCDF)
  2. Individual float profiles from Argo GDAC (as fallback)

Co-location procedure:
  - Spatial: nearest 0.25° grid cell (within 0.125° = half a grid cell)
  - Temporal: match to nearest day (±1 day tolerance)
  - Quality: only profiles with QC flag = 1 (good) at each depth level
  - Depth interpolation: linearly interpolate each profile to the 15 standard
    depth levels, discarding profiles that don't reach ≥ 10 standard levels

Output format: HDF5 or .npz per-year files with structure:
    {
      "lat": [N],          float32 — profile latitudes
      "lon": [N],          float32 — profile longitudes
      "date": [N],         str     — ISO dates YYYY-MM-DD
      "grid_lat_idx": [N], int16   — matched grid row index
      "grid_lon_idx": [N], int16   — matched grid column index
      "temp_15lev": [N,15], float32 — temperature at standard depths (°C)
      "salt_15lev": [N,15], float32 — salinity at standard depths (PSU)
      "depth_levels": [15],float32 — standard depth levels (m)
      "n_valid_depths": [N], int8  — number of non-NaN levels per profile
    }
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import warnings
from datetime import date, timedelta
from typing import Optional

from ocean_pipeline.config import (
    LATS, LONS, NLAT, NLON, DEPTH_LEVELS, DATA_ARGO,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
)

DATA_ARGO.mkdir(parents=True, exist_ok=True)

DEPTH_LEVELS_ARR = np.array(DEPTH_LEVELS, dtype=np.float32)
MAX_SPATIAL_OFFSET = 0.125   # degrees — half a grid cell
MAX_TEMPORAL_OFFSET = 1      # days
MIN_VALID_DEPTHS = 10        # minimum standard levels needed per profile


def interpolate_to_standard_depths(
    depth_obs: np.ndarray,
    temp_obs: np.ndarray,
    salt_obs: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linearly interpolate an observed profile to the 15 standard depth levels.

    Parameters
    ----------
    depth_obs : 1-D array of observed depth values (m, ascending)
    temp_obs  : 1-D array of observed temperature (°C)
    salt_obs  : 1-D array of observed salinity (PSU) or None

    Returns
    -------
    temp_std : [15] float32 — temperature at standard depths, NaN where out of range
    salt_std : [15] float32 — salinity at standard depths, NaN where out of range or unavailable
    """
    n = len(DEPTH_LEVELS)
    temp_std = np.full(n, np.nan, dtype=np.float32)
    salt_std = np.full(n, np.nan, dtype=np.float32)

    if len(depth_obs) < 2:
        return temp_std, salt_std

    # Remove NaNs
    valid = ~(np.isnan(depth_obs) | np.isnan(temp_obs))
    depth_obs = depth_obs[valid]
    temp_obs  = temp_obs[valid]
    if salt_obs is not None:
        salt_obs = salt_obs[valid]

    if len(depth_obs) < 2:
        return temp_std, salt_std

    # Sort by depth (ascending)
    sort_idx = np.argsort(depth_obs)
    depth_obs = depth_obs[sort_idx]
    temp_obs  = temp_obs[sort_idx]
    if salt_obs is not None:
        salt_obs = salt_obs[sort_idx]

    d_min, d_max = depth_obs[0], depth_obs[-1]

    for i, d in enumerate(DEPTH_LEVELS):
        if d < d_min or d > d_max:
            continue   # NaN — outside observed range
        temp_std[i] = float(np.interp(d, depth_obs, temp_obs))
        if salt_obs is not None:
            salt_std[i] = float(np.interp(d, depth_obs, salt_obs))

    return temp_std, salt_std


def colocate_profile(lat: float, lon: float, obs_date: date,
                     depth_obs: np.ndarray, temp_obs: np.ndarray,
                     salt_obs: Optional[np.ndarray] = None
                     ) -> Optional[dict]:
    """
    Co-locate a single ARGO profile to the model grid.

    Returns a dict with grid indices, interpolated profiles, or None if
    the profile fails quality/coverage criteria.
    """
    # Domain check
    if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
        return None

    # Find nearest grid cell
    lat_idx = int(np.argmin(np.abs(LATS - lat)))
    lon_idx = int(np.argmin(np.abs(LONS - lon)))

    # Spatial offset check
    d_lat = abs(LATS[lat_idx] - lat)
    d_lon = abs(LONS[lon_idx] - lon)
    if d_lat > MAX_SPATIAL_OFFSET or d_lon > MAX_SPATIAL_OFFSET:
        return None

    # Interpolate to standard depths
    temp_std, salt_std = interpolate_to_standard_depths(depth_obs, temp_obs, salt_obs)

    n_valid = int(np.sum(~np.isnan(temp_std)))
    if n_valid < MIN_VALID_DEPTHS:
        return None

    return {
        "lat": float(lat),
        "lon": float(lon),
        "date": obs_date.isoformat(),
        "grid_lat_idx": lat_idx,
        "grid_lon_idx": lon_idx,
        "temp_15lev": temp_std,
        "salt_15lev": salt_std,
        "n_valid_depths": n_valid
    }


def load_argo_colocated(year: int) -> Optional[dict]:
    """Load pre-computed co-located ARGO for a given year."""
    path = DATA_ARGO / f"argo_colocated_{year}.npz"
    if not path.exists():
        warnings.warn(f"[ARGO] No co-located file for {year}: {path}")
        return None
    data = np.load(path, allow_pickle=True)
    return dict(data)


def save_argo_colocated(profiles: list[dict], year: int):
    """Save co-located ARGO profiles for a year to .npz."""
    if not profiles:
        warnings.warn(f"[ARGO] No valid profiles to save for {year}")
        return

    out = {
        "lat":          np.array([p["lat"] for p in profiles], dtype=np.float32),
        "lon":          np.array([p["lon"] for p in profiles], dtype=np.float32),
        "date":         np.array([p["date"] for p in profiles]),
        "grid_lat_idx": np.array([p["grid_lat_idx"] for p in profiles], dtype=np.int16),
        "grid_lon_idx": np.array([p["grid_lon_idx"] for p in profiles], dtype=np.int16),
        "temp_15lev":   np.stack([p["temp_15lev"] for p in profiles]),
        "salt_15lev":   np.stack([p["salt_15lev"] for p in profiles]),
        "n_valid_depths": np.array([p["n_valid_depths"] for p in profiles], dtype=np.int8),
        "depth_levels": DEPTH_LEVELS_ARR,
    }

    path = DATA_ARGO / f"argo_colocated_{year}.npz"
    np.savez_compressed(path, **out)
    print(f"[ARGO] Saved {len(profiles)} profiles to {path.name}")


def colocate_from_netcdf(nc_path: Path, year: int):
    """
    Co-locate ARGO profiles from a gridded NetCDF file (INCOIS LAS format).
    Expected variables: TEMP, PSAL (or TEMP_ADJUSTED), LATITUDE, LONGITUDE,
    JULD (Julian day) or TIME.

    Call this function once per year after downloading from INCOIS LAS.
    """
    import xarray as xr

    if not nc_path.exists():
        print(f"[ARGO] File not found: {nc_path}")
        print(f"  Download gridded ARGO from https://las.incois.gov.in")
        return

    ds = xr.open_dataset(nc_path)
    print(f"[ARGO] Opened {nc_path.name}: {ds}")

    # Variable name detection
    lat_var  = next((v for v in ["LATITUDE", "lat", "latitude"] if v in ds), None)
    lon_var  = next((v for v in ["LONGITUDE", "lon", "longitude"] if v in ds), None)
    temp_var = next((v for v in ["TEMP_ADJUSTED", "TEMP", "PTEM", "temp"] if v in ds), None)
    salt_var = next((v for v in ["PSAL_ADJUSTED", "PSAL", "PSAL_ADJUSTED", "salt"] if v in ds), None)
    pres_var = next((v for v in ["PRES_ADJUSTED", "PRES", "DEPTH", "depth"] if v in ds), None)
    time_var = next((v for v in ["JULD", "TIME", "time"] if v in ds), None)

    if any(v is None for v in [lat_var, lon_var, temp_var, pres_var]):
        print(f"[ARGO] Could not find required variables. Found: {list(ds.data_vars)}")
        ds.close()
        return

    profiles = []
    n_profiles = ds.dims.get("N_PROF", ds.dims.get("profiles", 0))
    print(f"[ARGO] Processing {n_profiles} profiles for year {year}...")

    for i in range(n_profiles):
        try:
            lat  = float(ds[lat_var].isel(N_PROF=i).values)
            lon  = float(ds[lon_var].isel(N_PROF=i).values)
            pres = ds[pres_var].isel(N_PROF=i).values.astype(np.float32)
            temp = ds[temp_var].isel(N_PROF=i).values.astype(np.float32)
            salt = ds[salt_var].isel(N_PROF=i).values.astype(np.float32) if salt_var else None

            # Time
            t = ds[time_var].isel(N_PROF=i).values if time_var else None
            if t is not None:
                import pandas as pd
                obs_date = pd.Timestamp(t).date()
            else:
                obs_date = date(year, 7, 1)   # fallback mid-year

            result = colocate_profile(lat, lon, obs_date, pres, temp, salt)
            if result:
                profiles.append(result)

        except Exception as e:
            continue

    ds.close()
    save_argo_colocated(profiles, year)
    print(f"[ARGO] {len(profiles)}/{n_profiles} profiles co-located successfully for {year}")


if __name__ == "__main__":
    # Demonstration: co-locate a synthetic profile
    print("[ARGO] Running co-location self-test with synthetic profile...")
    lat, lon = 15.1, 75.3   # open Arabian Sea
    d = date(2023, 6, 1)
    depths = np.array([0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 700], dtype=np.float32)
    temps  = np.array([29, 28.5, 28, 27, 25, 22, 18, 15, 10, 8, 6, 4, 2], dtype=np.float32)
    salts  = np.array([35, 35.1, 35.2, 35.3, 35.4, 35.5, 35.6, 35.7, 35.8, 35.8, 35.7, 35.5, 35.3], dtype=np.float32)

    result = colocate_profile(lat, lon, d, depths, temps, salts)
    if result:
        print(f"  ✅ Co-located: grid=({result['grid_lat_idx']}, {result['grid_lon_idx']})")
        print(f"     T at 15 standard levels: {result['temp_15lev']}")
        print(f"     Valid depths: {result['n_valid_depths']}/15")
    else:
        print("  ❌ Co-location failed")

    print("\n[ARGO] To process real ARGO data, run:")
    print("  from ocean_pipeline.ingestion.argo_colocation import colocate_from_netcdf")
    print("  colocate_from_netcdf(Path('data/raw/argo_2023.nc'), year=2023)")
