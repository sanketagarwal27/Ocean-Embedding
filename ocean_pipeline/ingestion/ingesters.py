"""
Phase 0 - Step 2: Per-dataset ingestion modules.

Each ingester is independently runnable and testable on a small date range.
Handles: coordinate name normalization, domain subsetting, time selection,
and regridding to the common 0.25 deg daily grid.

Supported datasets:
    SST  -- OSTIA (METOFFICE-GLO-SST-L4-REP-OBS-SST)
    SSH  -- DUACS (c3s_obs-sl_glo_phy-ssh)
    SSS  -- SMAP/SMOS (cmems_obs-mob_glo_phy-sss)
    [MISSING] OSCAR currents (U, V)  -- needs PODAAC download
    [MISSING] CCMP winds (U, V)      -- needs PODAAC download
    [MISSING] GLORYS T, S            -- needs CMEMS download

NOTE on sample files:
    The NC files currently in the project are single-day snapshots used to
    validate the pipeline plumbing. Full multi-year downloads are required
    for training (see README.md). Call get_available_dates() on any ingester
    to find what dates are actually present in a given file.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import xarray as xr
import warnings
from datetime import datetime, date
from typing import Optional

from ocean_pipeline.config import (
    EXISTING_NC_FILES, LATS, LONS, NLAT, NLON,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, GRID_RES
)
from ocean_pipeline.preprocessing.regrid import regrid_to_target


# ---- Coordinate normalizer --------------------------------------------------
def _find_dim(ds: xr.Dataset, candidates: list) -> Optional[str]:
    for c in candidates:
        if c in ds.coords or c in ds.dims:
            return c
    return None


def _normalize_coords(ds: xr.Dataset) -> xr.Dataset:
    rename_map = {}
    lat_cand = ["lat", "latitude", "nav_lat", "yt_ocean", "y"]
    lon_cand = ["lon", "longitude", "nav_lon", "xt_ocean", "x"]
    dep_cand = ["depth", "level", "lev", "deptht", "z", "depth_below_sea"]

    lat = _find_dim(ds, lat_cand)
    lon = _find_dim(ds, lon_cand)
    dep = _find_dim(ds, dep_cand)

    if lat and lat != "latitude":  rename_map[lat] = "latitude"
    if lon and lon != "longitude": rename_map[lon] = "longitude"
    if dep and dep != "depth":     rename_map[dep] = "depth"

    if rename_map:
        ds = ds.rename(rename_map)
    return ds


def _subset_domain(ds: xr.Dataset) -> xr.Dataset:
    buf = 0.5
    lat_slice = slice(LAT_MIN - buf, LAT_MAX + buf)
    lon_slice = slice(LON_MIN - buf, LON_MAX + buf)
    if "latitude" in ds.coords:
        ds = ds.sel(latitude=lat_slice, longitude=lon_slice)
    return ds


def _select_time(ds: xr.Dataset,
                 date_range: Optional[tuple] = None) -> xr.Dataset:
    if date_range is None or "time" not in ds.coords:
        return ds
    return ds.sel(time=slice(*date_range))


# ============================================================================
#  Base ingester helper
# ============================================================================
class _BaseIngester:
    """Mixin that adds get_available_dates() to any ingester."""

    def _open_ds(self) -> xr.Dataset:
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        try:
            return xr.open_dataset(self.path, engine="netcdf4", chunks={})
        except Exception:
            return xr.open_dataset(self.path, engine="h5netcdf", chunks={})

    def get_available_dates(self) -> list:
        """Return list of datetime.date objects available in this file."""
        ds = self._open_ds()
        ds = _normalize_coords(ds)
        if "time" not in ds.coords:
            ds.close()
            return []
        times = ds.coords["time"].values
        ds.close()
        import pandas as pd
        return [pd.Timestamp(t).date() for t in times]


# ============================================================================
#  SST ingester -- OSTIA
# ============================================================================
class SSTIngester(_BaseIngester):
    """
    OSTIA SST (METOFFICE-GLO-SST-L4-REP-OBS-SST).
    Variable: analysed_sst (K) -> converted to degC.
    Native resolution: 0.05 deg, daily.
    """
    VARIABLE = "analysed_sst"
    UNIT_OFFSET = -273.15   # K -> C

    def __init__(self, path: Optional[Path] = None):
        self.path = path or EXISTING_NC_FILES["sst"]

    def load(self, date_range: Optional[tuple] = None) -> xr.Dataset:
        ds = self._open_ds()
        ds = _normalize_coords(ds)
        ds = _subset_domain(ds)
        ds = _select_time(ds, date_range)
        return ds

    def get_daily_field(self, date_range: Optional[tuple] = None) -> xr.DataArray:
        """Return regridded SST in degC, shape (time, NLAT, NLON)."""
        ds = self.load(date_range)
        da = ds[self.VARIABLE].astype(np.float32)

        if "mask" in ds:
            da = da.where(ds["mask"] == 1)
        if "quality_level" in ds:
            da = da.where(ds["quality_level"] >= 4)

        da = da + self.UNIT_OFFSET
        da.attrs["units"] = "degrees_C"
        da.attrs["long_name"] = "SST"

        if len(da.time) == 0:
            avail = self.get_available_dates()
            warnings.warn(
                f"[SST] No data for date_range={date_range}. "
                f"Available dates in file: {avail}. "
                f"Download the full time series for training."
            )
            return da

        print(f"  [SST] Regridding {len(da.time)} time step(s) to {GRID_RES}deg grid...")
        slices = []
        for t in da.time.values:
            sl = da.sel(time=t)
            rg = regrid_to_target(sl, LATS, LONS,
                                  lat_dim="latitude", lon_dim="longitude")
            slices.append(rg)

        out = xr.concat(slices, dim=da.time)
        out.name = "sst"
        ds.close()
        return out


# ============================================================================
#  SSH ingester -- DUACS
# ============================================================================
class SSHIngester(_BaseIngester):
    """
    DUACS SSH/SLA (c3s_obs-sl_glo_phy-ssh).
    Variables: adt (absolute dynamic topography, m) -- preferred; sla fallback.
    Native resolution: 0.25 deg, daily.
    Also exposes geostrophic current components (ugos, vgos) if present.
    """
    VARIABLE = "adt"

    def __init__(self, path: Optional[Path] = None):
        self.path = path or EXISTING_NC_FILES["ssh"]

    def load(self, date_range: Optional[tuple] = None) -> xr.Dataset:
        ds = self._open_ds()
        ds = _normalize_coords(ds)
        ds = _subset_domain(ds)
        ds = _select_time(ds, date_range)
        return ds

    def get_daily_field(self, date_range: Optional[tuple] = None) -> xr.DataArray:
        """Return SSH/ADT in metres, shape (time, NLAT, NLON)."""
        ds = self.load(date_range)

        var = self.VARIABLE if self.VARIABLE in ds else "sla"
        if var not in ds:
            var = list(ds.data_vars)[0]
            warnings.warn(f"[SSH] Expected adt/sla, using '{var}'")

        da = ds[var].astype(np.float32)
        da.attrs["units"] = "m"
        da.attrs["long_name"] = "SSH"

        if len(da.time) == 0:
            avail = self.get_available_dates()
            warnings.warn(
                f"[SSH] No data for date_range={date_range}. "
                f"Available dates: {avail}"
            )
            return da

        print(f"  [SSH] Regridding {len(da.time)} time step(s)...")
        slices = []
        for t in da.time.values:
            sl = da.sel(time=t)
            rg = regrid_to_target(sl, LATS, LONS,
                                  lat_dim="latitude", lon_dim="longitude")
            slices.append(rg)

        out = xr.concat(slices, dim=da.time)
        out.name = "ssh"
        ds.close()
        return out

    def get_geostrophic_currents(self, date_range: Optional[tuple] = None):
        """
        Return (ugos, vgos) geostrophic current components from DUACS if present.
        These can substitute for OSCAR until OSCAR is downloaded.
        Shape: (time, NLAT, NLON) each.
        """
        ds = self.load(date_range)
        u_var = "ugos" if "ugos" in ds else None
        v_var = "vgos" if "vgos" in ds else None

        if u_var is None or v_var is None:
            warnings.warn("[SSH] ugos/vgos not found in SSH file")
            return None, None

        if len(ds[u_var].time) == 0:
            return None, None

        print(f"  [SSH/geo] Regridding geostrophic currents {len(ds[u_var].time)} step(s)...")
        u_slices, v_slices = [], []
        for t in ds[u_var].time.values:
            u_sl = ds[u_var].sel(time=t).astype(np.float32)
            v_sl = ds[v_var].sel(time=t).astype(np.float32)
            u_slices.append(regrid_to_target(u_sl, LATS, LONS))
            v_slices.append(regrid_to_target(v_sl, LATS, LONS))

        u_out = xr.concat(u_slices, dim=ds[u_var].time)
        v_out = xr.concat(v_slices, dim=ds[v_var].time)
        u_out.name = "u_current"
        v_out.name = "v_current"
        ds.close()
        return u_out, v_out


# ============================================================================
#  SSS ingester -- SMAP/SMOS (CMEMS multi-obs)
# ============================================================================
class SSSIngester(_BaseIngester):
    """
    CMEMS Multi-obs SSS (cmems_obs-mob_glo_phy-sss_my_multi_P1D).
    Variable: sos (sea_surface_salinity, PSU).
    Native resolution: 0.125 deg, daily.
    """
    VARIABLE = "sos"

    def __init__(self, path: Optional[Path] = None):
        self.path = path or EXISTING_NC_FILES["sss"]

    def load(self, date_range: Optional[tuple] = None) -> xr.Dataset:
        ds = self._open_ds()
        ds = _normalize_coords(ds)
        ds = _subset_domain(ds)
        ds = _select_time(ds, date_range)
        return ds

    def get_daily_field(self, date_range: Optional[tuple] = None) -> xr.DataArray:
        """Return SSS in PSU, shape (time, NLAT, NLON)."""
        ds = self.load(date_range)
        var = self.VARIABLE if self.VARIABLE in ds else list(ds.data_vars)[0]
        da = ds[var].astype(np.float32)

        # Drop any extra dimensions (e.g. depth=-0 in SSS file)
        for extra_dim in list(da.dims):
            if extra_dim not in ("time", "latitude", "longitude"):
                da = da.isel({extra_dim: 0})

        da.attrs["units"] = "PSU"
        da.attrs["long_name"] = "SSS"

        if len(da.time) == 0:
            avail = self.get_available_dates()
            warnings.warn(
                f"[SSS] No data for date_range={date_range}. "
                f"Available dates: {avail}"
            )
            return da

        print(f"  [SSS] Regridding {len(da.time)} time step(s)...")
        slices = []
        for t in da.time.values:
            sl = da.sel(time=t)
            rg = regrid_to_target(sl, LATS, LONS,
                                  lat_dim="latitude", lon_dim="longitude")
            slices.append(rg)

        out = xr.concat(slices, dim=da.time)
        out.name = "sss"
        ds.close()
        return out


# ============================================================================
#  Stub ingesters for NOT-YET-DOWNLOADED datasets
# ============================================================================
class OSCARIngester:
    """
    OSCAR L4 surface currents (U, V).
    Source: https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_FINAL_V2.0
    Status: NOT YET DOWNLOADED -- stub returns NaN placeholder arrays.

    FALLBACK: SSHIngester.get_geostrophic_currents() returns DUACS ugos/vgos
    which are a reasonable proxy for surface currents until OSCAR is available.
    """
    def get_daily_field(self, date_range=None):
        warnings.warn(
            "[OSCAR] Data not downloaded -- returning NaN placeholder. "
            "Use SSHIngester.get_geostrophic_currents() as fallback.",
            stacklevel=2
        )
        shape = (1, NLAT, NLON)
        u = xr.DataArray(np.full(shape, np.nan, dtype=np.float32), name="u_current",
                         dims=["time", "latitude", "longitude"])
        v = xr.DataArray(np.full(shape, np.nan, dtype=np.float32), name="v_current",
                         dims=["time", "latitude", "longitude"])
        return u, v


class CCMPIngester:
    """
    CCMP V3.1 / ASCAT surface winds (U, V at 10m).
    Source: https://podaac.jpl.nasa.gov/dataset/CCMP_WINDS_10M6HR_L4_V3.1
    Status: NOT YET DOWNLOADED -- stub returns NaN placeholder arrays.
    """
    def get_daily_field(self, date_range=None):
        warnings.warn(
            "[CCMP] Data not downloaded -- returning NaN placeholder.",
            stacklevel=2
        )
        shape = (1, NLAT, NLON)
        u = xr.DataArray(np.full(shape, np.nan, dtype=np.float32), name="u_wind",
                         dims=["time", "latitude", "longitude"])
        v = xr.DataArray(np.full(shape, np.nan, dtype=np.float32), name="v_wind",
                         dims=["time", "latitude", "longitude"])
        return u, v


class GLORYSIngester:
    """
    GLORYS Global Ocean Reanalysis (temperature + salinity, 3-D).
    Source: https://doi.org/10.48670/moi-00021
    Status: NOT YET DOWNLOADED -- stub returns NaN placeholder.
    """
    DEPTH_LEVELS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

    def __init__(self, path: Optional[Path] = None):
        self.path = path

    def get_daily_field(self, date_range=None):
        if self.path is None or not self.path.exists():
            warnings.warn(
                "[GLORYS] Data not downloaded -- returning NaN placeholder. "
                "Download from https://doi.org/10.48670/moi-00021",
                stacklevel=2
            )
            shape = (1, len(self.DEPTH_LEVELS), NLAT, NLON)
            temp = xr.DataArray(
                np.full(shape, np.nan, dtype=np.float32), name="thetao",
                dims=["time", "depth", "latitude", "longitude"],
                coords={"depth": self.DEPTH_LEVELS}
            )
            salt = xr.DataArray(
                np.full(shape, np.nan, dtype=np.float32), name="so",
                dims=["time", "depth", "latitude", "longitude"],
                coords={"depth": self.DEPTH_LEVELS}
            )
            return temp, salt
        raise NotImplementedError("GLORYSIngester real-data path not yet implemented")


# ============================================================================
#  Quick test
# ============================================================================
if __name__ == "__main__":
    print("\n[ingest] Querying available dates in each file...")
    for cls, name in [(SSTIngester, "SST"), (SSHIngester, "SSH"), (SSSIngester, "SSS")]:
        try:
            ing = cls()
            avail = ing.get_available_dates()
            print(f"  {name}: {avail}")
        except Exception as e:
            print(f"  {name}: ERROR {e}")

    print("\n[ingest] Testing ingesters on available dates...")
    for cls, name in [(SSTIngester, "SST"), (SSHIngester, "SSH"), (SSSIngester, "SSS")]:
        try:
            ing = cls()
            avail = ing.get_available_dates()
            if not avail:
                print(f"  [SKIP] {name}: no dates in file")
                continue
            # Use the actual available date
            d_str = avail[0].strftime("%Y-%m-%d")
            da = ing.get_daily_field(date_range=(d_str, d_str))
            if len(da.time) > 0:
                print(f"  [OK] {name}: shape={da.shape}, "
                      f"mean={float(np.nanmean(da.values)):.3f}, "
                      f"NaN%={100*np.isnan(da.values).mean():.1f}%")
            else:
                print(f"  [WARN] {name}: 0 time steps returned for {d_str}")
        except Exception as e:
            import traceback
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()

    print("\n[ingest] SSH geostrophic currents (OSCAR fallback)...")
    try:
        ing = SSHIngester()
        avail = ing.get_available_dates()
        if avail:
            d_str = avail[0].strftime("%Y-%m-%d")
            u, v = ing.get_geostrophic_currents(date_range=(d_str, d_str))
            if u is not None:
                print(f"  [OK] ugos: shape={u.shape}, "
                      f"mean={float(np.nanmean(u.values)):.3f} m/s")
                print(f"  [OK] vgos: shape={v.shape}, "
                      f"mean={float(np.nanmean(v.values)):.3f} m/s")
    except Exception as e:
        print(f"  [FAIL]: {e}")
