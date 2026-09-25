"""
Phase 0 - Step 1: Inspect and catalogue existing raw NetCDF files.

Run with:
    .\\venv\\Scripts\\python.exe ocean_pipeline\\ingestion\\inspect_raw_files.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import xarray as xr
import numpy as np

from ocean_pipeline.config import EXISTING_NC_FILES, LOGS_DIR, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX


def inspect_file(label: str, path: Path) -> dict:
    """Open a NetCDF file and return key metadata."""
    print(f"\n{'='*60}")
    print(f"  [{label.upper()}]  {path.name}")
    print(f"{'='*60}")

    if not path.exists():
        print(f"  [NOT FOUND] FILE NOT FOUND: {path}")
        return {"label": label, "found": False}

    try:
        ds = xr.open_dataset(path, engine="netcdf4", chunks={})
    except Exception as e:
        try:
            ds = xr.open_dataset(path, engine="h5netcdf", chunks={})
        except Exception as e2:
            print(f"  [ERROR] Could not open: {e} | {e2}")
            return {"label": label, "found": True, "error": str(e2)}

    print(f"\n  [Variables]")
    for vname, var in ds.data_vars.items():
        print(f"      {vname:30s}  shape={var.shape}  units={var.attrs.get('units','?')}")

    print(f"\n  [Coordinates]")
    for cname, coord in ds.coords.items():
        if len(coord) > 4:
            print(f"      {cname:20s}  n={len(coord)}  "
                  f"[{coord.values[0]} -> {coord.values[-1]}]")
        else:
            print(f"      {cname:20s}  values={coord.values}")

    # Time range
    if "time" in ds.coords:
        t = ds.coords["time"].values
        print(f"\n  [Time] {np.datetime_as_string(t[0], unit='D')} -> "
              f"{np.datetime_as_string(t[-1], unit='D')}  (n={len(t)})")

    # Spatial extent
    lat_name = next((c for c in ds.coords if c.lower() in ("lat", "latitude")), None)
    lon_name = next((c for c in ds.coords if c.lower() in ("lon", "longitude")), None)
    if lat_name and lon_name:
        lats = ds.coords[lat_name].values
        lons = ds.coords[lon_name].values
        print(f"\n  [Spatial] lat=[{lats.min():.2f}, {lats.max():.2f}]  "
              f"lon=[{lons.min():.2f}, {lons.max():.2f}]")
        print(f"      Native res ~ {abs(np.diff(lats).mean()):.4f}deg x "
              f"{abs(np.diff(lons).mean()):.4f}deg")
        in_domain_lat = ((lats >= LAT_MIN) & (lats <= LAT_MAX)).sum()
        in_domain_lon = ((lons >= LON_MIN) & (lons <= LON_MAX)).sum()
        print(f"      Domain overlap: {in_domain_lat} lat cells x {in_domain_lon} lon cells")

    print(f"\n  [Attributes (first 6)]")
    for k, v in list(ds.attrs.items())[:6]:
        print(f"      {k}: {str(v)[:80]}")

    ds.close()
    return {"label": label, "found": True, "path": str(path)}


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print("  OCEAN EMBEDDING -- Raw Data Inventory")
    print(f"  Target domain: lat=[{LAT_MIN}, {LAT_MAX}]  lon=[{LON_MIN}, {LON_MAX}]")
    print(f"{'#'*60}")

    results = []
    for label, path in EXISTING_NC_FILES.items():
        result = inspect_file(label, path)
        results.append(result)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    missing = [r["label"] for r in results if not r.get("found", False)]
    if missing:
        print(f"  [WARN] MISSING files: {missing}")
    else:
        print(f"  [OK] All {len(results)} files found.")

    print(f"\n  [Missing datasets (not yet downloaded)]")
    print(f"      OSCAR currents (U, V)       -> PODAAC")
    print(f"      CCMP/ASCAT winds (U, V)     -> PODAAC")
    print(f"      GLORYS reanalysis (T, S)    -> CMEMS doi:10.48670/moi-00021")
    print(f"      ARGO gridded profiles       -> INCOIS LAS")
    print(f"\n  See README.md for download instructions.")


if __name__ == "__main__":
    main()
