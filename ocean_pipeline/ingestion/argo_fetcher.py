"""
ARGO profile fetcher using the argopy library.

Downloads individual Argo float profiles for the North Indian Ocean,
co-locates them to the model grid, and saves as held-out validation data.

The ARGO validation set is NEVER used during training. It provides
independent ground-truth subsurface T/S profiles for skill evaluation.

Prerequisites:
    pip install argopy

Usage:
    from ocean_pipeline.ingestion.argo_fetcher import fetch_argo_profiles
    fetch_argo_profiles(year=2023)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import warnings
from datetime import date
from typing import Optional

from ocean_pipeline.config import (
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    DEPTH_LEVELS, DATA_ARGO, DATA_RAW
)
from ocean_pipeline.ingestion.argo_colocation import (
    colocate_profile, save_argo_colocated, DEPTH_LEVELS_ARR
)


def fetch_argo_profiles(
    year: int,
    month: Optional[int] = None,
    force: bool = False,
) -> Path:
    """
    Fetch ARGO profiles from the Argo GDAC via argopy and co-locate to grid.

    Parameters
    ----------
    year  : year to fetch
    month : optional specific month (1-12); if None, fetches full year
    force : re-fetch even if co-located file exists

    Returns
    -------
    Path to the saved co-located .npz file.
    """
    out_path = DATA_ARGO / f"argo_colocated_{year}.npz"
    if out_path.exists() and not force:
        print(f"  [ARGO] SKIP (exists): {out_path.name}")
        return out_path

    try:
        import argopy
    except ImportError:
        print("  [ARGO] argopy not installed. Install with: pip install argopy")
        print("  [ARGO] Alternatively, download ARGO NetCDF from INCOIS LAS")
        print("         and use argo_colocation.colocate_from_netcdf()")
        return out_path

    print(f"  [ARGO] Fetching profiles for {year} from Argo GDAC...")

    # Time bounds
    if month:
        import calendar
        last_day = calendar.monthrange(year, month)[1]
        start = f"{year}-{month:02d}-01"
        end = f"{year}-{month:02d}-{last_day:02d}"
    else:
        start = f"{year}-01-01"
        end = f"{year}-12-31"

    try:
        # Use argopy DataFetcher to get profiles in the NIO domain
        fetcher = argopy.DataFetcher(src="gdac", mode="standard")
        ds = fetcher.region(
            [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, 0, 1100, start, end]
        ).to_xarray()

        print(f"  [ARGO] Retrieved {len(ds.N_PROF) if 'N_PROF' in ds.dims else '?'} "
              f"profiles for {year}")

    except Exception as e:
        print(f"  [ARGO] Failed to fetch from GDAC: {e}")
        print(f"  [ARGO] Trying alternative source (erddap)...")

        try:
            fetcher = argopy.DataFetcher(src="erddap", mode="standard")
            ds = fetcher.region(
                [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, 0, 1100, start, end]
            ).to_xarray()
        except Exception as e2:
            print(f"  [ARGO] All sources failed: {e2}")
            print(f"  [ARGO] Download manually from INCOIS LAS:")
            print(f"         https://las.incois.gov.in")
            return out_path

    # Co-locate profiles to the model grid
    profiles = []
    n_profiles = ds.dims.get("N_PROF", 0)

    # Find variable names
    lat_var = next((v for v in ["LATITUDE", "latitude", "lat"] if v in ds), None)
    lon_var = next((v for v in ["LONGITUDE", "longitude", "lon"] if v in ds), None)
    temp_var = next((v for v in ["TEMP_ADJUSTED", "TEMP", "temp"] if v in ds), None)
    salt_var = next((v for v in ["PSAL_ADJUSTED", "PSAL", "psal"] if v in ds), None)
    pres_var = next((v for v in ["PRES_ADJUSTED", "PRES", "pres"] if v in ds), None)
    time_var = next((v for v in ["TIME", "JULD", "time"] if v in ds), None)

    if any(v is None for v in [lat_var, lon_var, temp_var, pres_var]):
        print(f"  [ARGO] Could not find required variables. Found: {list(ds.data_vars)}")
        return out_path

    print(f"  [ARGO] Co-locating {n_profiles} profiles...")

    for i in range(n_profiles):
        try:
            import pandas as pd

            lat = float(ds[lat_var].isel(N_PROF=i).values)
            lon = float(ds[lon_var].isel(N_PROF=i).values)
            pres = ds[pres_var].isel(N_PROF=i).values.astype(np.float32)
            temp = ds[temp_var].isel(N_PROF=i).values.astype(np.float32)
            salt = ds[salt_var].isel(N_PROF=i).values.astype(np.float32) if salt_var else None

            if time_var:
                t = ds[time_var].isel(N_PROF=i).values
                obs_date = pd.Timestamp(t).date()
            else:
                obs_date = date(year, 7, 1)

            result = colocate_profile(lat, lon, obs_date, pres, temp, salt)
            if result:
                profiles.append(result)

        except Exception:
            continue

    if profiles:
        save_argo_colocated(profiles, year)
        print(f"  [ARGO] {len(profiles)}/{n_profiles} profiles co-located for {year}")
    else:
        print(f"  [ARGO] No valid profiles co-located for {year}")

    return out_path


def fetch_argo_all(years: range, force: bool = False) -> list[Path]:
    """Fetch ARGO for multiple years."""
    paths = []
    for year in years:
        p = fetch_argo_profiles(year, force=force)
        paths.append(p)
    return paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch ARGO validation data")
    parser.add_argument("--years", type=str, default="2015-2023")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if "-" in args.years:
        s, e = args.years.split("-")
        years = range(int(s), int(e) + 1)
    else:
        years = range(int(args.years), int(args.years) + 1)

    fetch_argo_all(years, force=args.force)
