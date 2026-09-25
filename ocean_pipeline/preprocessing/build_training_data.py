"""
Build aligned, regridded training tensors from downloaded CMEMS data.

Reads raw NetCDF files from data/raw/cmems/<variable>/, regrids to
the common 0.25deg grid, aligns all variables to the same daily timeline,
and saves compressed .npz tensors to data/processed/training/.

Each daily .npz file contains:
    input  : [NLAT=101, NLON=241, 7]  float32  — surface channels
    target : [NLAT=101, NLON=241, 15] float32  — GLORYS T at 15 depth levels
    target_salt : [NLAT=101, NLON=241, 15] float32  — GLORYS S at 15 depths
    date   : str  — ISO date

After processing all years, normalization statistics (mean, std per channel)
are computed from the TRAINING years only and saved to data/processed/norm_stats.npz.

Usage:
    from ocean_pipeline.preprocessing.build_training_data import TrainingDataBuilder
    builder = TrainingDataBuilder()
    builder.build_all(years=range(2015, 2024))
    builder.compute_norm_stats()
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import xarray as xr
import warnings
from datetime import date, timedelta
from typing import Optional
import traceback

from ocean_pipeline.config import (
    LATS, LONS, NLAT, NLON, NCHAN, NDEPTH,
    INPUT_CHANNELS, DEPTH_LEVELS, DATA_RAW, DATA_PROCESSED, DATA_MASKS,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, GRID_RES,
    TRAIN_YEARS, VAL_YEARS, TEST_YEARS
)
from ocean_pipeline.preprocessing.regrid import regrid_to_target


# Standard depth levels for GLORYS interpolation
DEPTH_LEVELS_ARR = np.array(DEPTH_LEVELS, dtype=np.float32)


def _normalize_coords(ds):
    """Rename coordinates to standard names (latitude, longitude, time, depth)."""
    rename_map = {}
    for src, tgt in [
        (["lat", "nav_lat", "y"], "latitude"),
        (["lon", "nav_lon", "x"], "longitude"),
        (["lev", "level", "deptht", "z"], "depth"),
    ]:
        for s in src:
            if s in ds.coords or s in ds.dims:
                rename_map[s] = tgt
                break
    if rename_map:
        ds = ds.rename(rename_map)
    return ds


def _interp_to_standard_depths(data_3d, src_depths):
    """
    Interpolate a 3D field [depth, lat, lon] from src_depths to DEPTH_LEVELS.

    Parameters
    ----------
    data_3d   : np.ndarray [D_src, H, W]
    src_depths: 1D array of source depth values

    Returns
    -------
    np.ndarray [15, H, W] interpolated to standard depth levels
    """
    from scipy.interpolate import interp1d

    D, H, W = data_3d.shape
    result = np.full((NDEPTH, H, W), np.nan, dtype=np.float32)

    src_depths = np.array(src_depths, dtype=np.float64)

    # Vectorized interpolation along depth axis
    for i in range(H):
        for j in range(W):
            col = data_3d[:, i, j].astype(np.float64)
            valid = ~np.isnan(col)
            if valid.sum() < 2:
                continue
            try:
                f = interp1d(
                    src_depths[valid], col[valid],
                    kind="linear", bounds_error=False, fill_value=(col[valid][0], col[valid][-1])
                )
                result[:, i, j] = f(DEPTH_LEVELS_ARR).astype(np.float32)
            except Exception:
                continue

    return result


class TrainingDataBuilder:
    """
    Build aligned daily training tensors from downloaded CMECS NetCDF files.
    """

    def __init__(self,
                 cmecs_dir: Optional[Path] = None,
                 output_dir: Optional[Path] = None,
                 mask_path: Optional[Path] = None):
        self.cmecs_dir = cmecs_dir or (DATA_RAW / "cmems")
        self.output_dir = output_dir or (DATA_PROCESSED / "training")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load ocean mask if available
        mp = mask_path or (DATA_MASKS / "ocean_mask.npy")
        self.ocean_mask = np.load(mp) if mp.exists() else None

    # ------------------------------------------------------------------
    #  Open and cache dataset handles
    # ------------------------------------------------------------------
    def _open_yearly_ds(self, var_key: str, year: int) -> Optional[xr.Dataset]:
        """Open a yearly NetCDF file (sst, ssh, sss, wind)."""
        path = self.cmecs_dir / var_key / f"{var_key}_{year}.nc"
        if not path.exists():
            return None
        try:
            ds = xr.open_dataset(path, engine="netcdf4", chunks={})
        except Exception:
            try:
                ds = xr.open_dataset(path, engine="h5netcdf", chunks={})
            except Exception as e:
                warnings.warn(f"Cannot open {path}: {e}")
                return None
        return _normalize_coords(ds)

    def _open_glorys_month(self, year: int, month: int) -> Optional[xr.Dataset]:
        """Open a monthly GLORYS NetCDF file."""
        path = self.cmecs_dir / "glorys" / f"glorys_{year}_{month:02d}.nc"
        if not path.exists():
            return None
        try:
            ds = xr.open_dataset(path, engine="netcdf4", chunks={})
        except Exception:
            try:
                ds = xr.open_dataset(path, engine="h5netcdf", chunks={})
            except Exception as e:
                warnings.warn(f"Cannot open {path}: {e}")
                return None
        return _normalize_coords(ds)

    # ------------------------------------------------------------------
    #  Extract a single-day 2D field from a dataset
    # ------------------------------------------------------------------
    def _extract_day_2d(self, ds: xr.Dataset, var_name: str,
                        target_date: date) -> Optional[np.ndarray]:
        """
        Extract and regrid a single-day 2D field to [NLAT, NLON].

        Returns None if variable or date not found.
        """
        if var_name not in ds:
            return None

        da = ds[var_name]

        # Drop extra dims (e.g. depth=-0 in SSS)
        for dim in list(da.dims):
            if dim not in ("time", "latitude", "longitude"):
                da = da.isel({dim: 0})

        # Select time
        import pandas as pd
        target_ts = pd.Timestamp(target_date)
        try:
            sl = da.sel(time=target_ts, method="nearest",
                        tolerance=pd.Timedelta("1D"))
        except (KeyError, ValueError):
            return None

        # Regrid
        sl_2d = sl.astype(np.float32)
        try:
            rg = regrid_to_target(sl_2d, LATS, LONS,
                                  lat_dim="latitude", lon_dim="longitude")
            return rg.values
        except Exception:
            return None

    def _extract_glorys_day(self, ds: xr.Dataset, target_date: date
                           ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract GLORYS T and S profiles for one day, regridded and interpolated
        to standard depth levels.

        Returns (temp_15lev, salt_15lev) each [15, NLAT, NLON], or (None, None).
        """
        import pandas as pd
        target_ts = pd.Timestamp(target_date)

        temp_out = None
        salt_out = None

        for var_name, label in [("thetao", "temp"), ("so", "salt")]:
            if var_name not in ds:
                continue

            da = ds[var_name]

            # Select time
            try:
                da_day = da.sel(time=target_ts, method="nearest",
                               tolerance=pd.Timedelta("1D"))
            except (KeyError, ValueError):
                continue

            # Get source depths
            if "depth" in da_day.dims:
                src_depths = da_day.depth.values
            else:
                continue

            # Regrid each depth level
            n_depths = len(src_depths)
            regridded = np.full((n_depths, NLAT, NLON), np.nan, dtype=np.float32)

            for d_idx in range(n_depths):
                sl = da_day.isel(depth=d_idx).astype(np.float32)
                try:
                    rg = regrid_to_target(sl, LATS, LONS,
                                          lat_dim="latitude", lon_dim="longitude")
                    regridded[d_idx] = rg.values
                except Exception:
                    continue

            # Interpolate to standard 15 depth levels
            interpolated = _interp_to_standard_depths(regridded, src_depths)

            if label == "temp":
                temp_out = interpolated
            else:
                salt_out = interpolated

        return temp_out, salt_out

    # ------------------------------------------------------------------
    #  Build one day's tensor
    # ------------------------------------------------------------------
    def build_day(self, target_date: date,
                  ds_sst: Optional[xr.Dataset] = None,
                  ds_ssh: Optional[xr.Dataset] = None,
                  ds_sss: Optional[xr.Dataset] = None,
                  ds_wind: Optional[xr.Dataset] = None,
                  ds_glorys: Optional[xr.Dataset] = None,
                  save: bool = True) -> tuple:
        """
        Build a single day's input [H, W, 7] and target [H, W, 15] tensors.

        Returns (input_tensor, target_temp, target_salt, channel_status)
        """
        input_tensor = np.full((NLAT, NLON, NCHAN), np.nan, dtype=np.float32)
        target_temp = np.full((NLAT, NLON, NDEPTH), np.nan, dtype=np.float32)
        target_salt = np.full((NLAT, NLON, NDEPTH), np.nan, dtype=np.float32)
        status = {}

        # SST
        if ds_sst is not None:
            field = self._extract_day_2d(ds_sst, "analysed_sst", target_date)
            if field is not None:
                # Convert K to degC if values suggest Kelvin
                if np.nanmean(field) > 200:
                    field = field - 273.15
                input_tensor[:, :, INPUT_CHANNELS.index("sst")] = field
                status["sst"] = "OK"
            else:
                status["sst"] = "NO_DATE"
        else:
            status["sst"] = "NO_FILE"

        # SSH (ADT)
        if ds_ssh is not None:
            field = self._extract_day_2d(ds_ssh, "adt", target_date)
            if field is None:
                field = self._extract_day_2d(ds_ssh, "sla", target_date)
            if field is not None:
                input_tensor[:, :, INPUT_CHANNELS.index("ssh")] = field
                status["ssh"] = "OK"
            else:
                status["ssh"] = "NO_DATE"

            # Geostrophic currents from DUACS
            u_geo = self._extract_day_2d(ds_ssh, "ugos", target_date)
            v_geo = self._extract_day_2d(ds_ssh, "vgos", target_date)
            if u_geo is not None:
                input_tensor[:, :, INPUT_CHANNELS.index("u_current")] = u_geo
                input_tensor[:, :, INPUT_CHANNELS.index("v_current")] = v_geo
                status["u_current"] = "OK(geo)"
                status["v_current"] = "OK(geo)"
            else:
                status["u_current"] = "NaN"
                status["v_current"] = "NaN"
        else:
            status["ssh"] = "NO_FILE"
            status["u_current"] = "NO_FILE"
            status["v_current"] = "NO_FILE"

        # SSS
        if ds_sss is not None:
            field = self._extract_day_2d(ds_sss, "sos", target_date)
            if field is not None:
                input_tensor[:, :, INPUT_CHANNELS.index("sss")] = field
                status["sss"] = "OK"
            else:
                status["sss"] = "NO_DATE"
        else:
            status["sss"] = "NO_FILE"

        # Wind
        if ds_wind is not None:
            u_wind = self._extract_day_2d(ds_wind, "eastward_wind", target_date)
            v_wind = self._extract_day_2d(ds_wind, "northward_wind", target_date)
            if u_wind is not None:
                input_tensor[:, :, INPUT_CHANNELS.index("u_wind")] = u_wind
                input_tensor[:, :, INPUT_CHANNELS.index("v_wind")] = v_wind
                status["u_wind"] = "OK"
                status["v_wind"] = "OK"
            else:
                status["u_wind"] = "NO_DATE"
                status["v_wind"] = "NO_DATE"
        else:
            status["u_wind"] = "NaN"
            status["v_wind"] = "NaN"

        # GLORYS target
        if ds_glorys is not None:
            temp_3d, salt_3d = self._extract_glorys_day(ds_glorys, target_date)
            if temp_3d is not None:
                # Transpose from [15, H, W] to [H, W, 15]
                target_temp = np.transpose(temp_3d, (1, 2, 0))
                status["glorys_T"] = "OK"
            else:
                status["glorys_T"] = "NO_DATE"
            if salt_3d is not None:
                target_salt = np.transpose(salt_3d, (1, 2, 0))
                status["glorys_S"] = "OK"
            else:
                status["glorys_S"] = "NO_DATE"
        else:
            status["glorys_T"] = "NO_FILE"
            status["glorys_S"] = "NO_FILE"

        # Apply ocean mask
        if self.ocean_mask is not None:
            land = ~self.ocean_mask
            input_tensor[land, :] = np.nan
            target_temp[land, :] = np.nan
            target_salt[land, :] = np.nan

        # Save
        if save:
            year_dir = self.output_dir / str(target_date.year)
            year_dir.mkdir(parents=True, exist_ok=True)
            out_path = year_dir / f"{target_date.isoformat()}.npz"
            np.savez_compressed(
                out_path,
                input=input_tensor,
                target=target_temp,
                target_salt=target_salt,
                date=np.array([target_date.isoformat()]),
                channels=np.array(INPUT_CHANNELS),
                depths=np.array(DEPTH_LEVELS),
                status=np.array(list(status.items())),
            )

        return input_tensor, target_temp, target_salt, status

    # ------------------------------------------------------------------
    #  Build all days for one year
    # ------------------------------------------------------------------
    def build_year(self, year: int, force: bool = False) -> int:
        """
        Build training tensors for all days in a year.

        Opens yearly surface obs files and monthly GLORYS files,
        iterates over each day, builds and saves tensors.

        Returns number of days processed.
        """
        import calendar
        from tqdm import tqdm

        year_dir = self.output_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        # Open yearly surface obs datasets
        ds_sst = self._open_yearly_ds("sst", year)
        ds_ssh = self._open_yearly_ds("ssh", year)
        ds_sss = self._open_yearly_ds("sss", year)
        ds_wind = self._open_yearly_ds("wind", year)

        print(f"\n  [build] Year {year}:")
        print(f"    SST:  {'loaded' if ds_sst else 'MISSING'}")
        print(f"    SSH:  {'loaded' if ds_ssh else 'MISSING'}")
        print(f"    SSS:  {'loaded' if ds_sss else 'MISSING'}")
        print(f"    Wind: {'loaded' if ds_wind else 'MISSING'}")

        n_days_in_year = 366 if calendar.isleap(year) else 365
        count = 0

        # Process month by month (GLORYS is monthly)
        for month in range(1, 13):
            ds_glorys = self._open_glorys_month(year, month)
            last_day = calendar.monthrange(year, month)[1]

            glorys_status = "loaded" if ds_glorys else "MISSING"
            print(f"    GLORYS {year}-{month:02d}: {glorys_status}")

            for day in range(1, last_day + 1):
                target_date = date(year, month, day)
                out_path = year_dir / f"{target_date.isoformat()}.npz"

                if out_path.exists() and not force:
                    count += 1
                    continue

                try:
                    self.build_day(
                        target_date,
                        ds_sst=ds_sst,
                        ds_ssh=ds_ssh,
                        ds_sss=ds_sss,
                        ds_wind=ds_wind,
                        ds_glorys=ds_glorys,
                        save=True,
                    )
                    count += 1
                except Exception as e:
                    print(f"    FAILED: {target_date}: {e}")

            if ds_glorys:
                ds_glorys.close()

        # Close yearly datasets
        for ds in [ds_sst, ds_ssh, ds_sss, ds_wind]:
            if ds is not None:
                ds.close()

        print(f"    Processed {count}/{n_days_in_year} days for {year}")
        return count

    # ------------------------------------------------------------------
    #  Build all years
    # ------------------------------------------------------------------
    def build_all(self, years: Optional[range] = None, force: bool = False):
        """Build training tensors for all specified years."""
        if years is None:
            all_years = sorted(set(TRAIN_YEARS + VAL_YEARS + TEST_YEARS))
            years = range(min(all_years), max(all_years) + 1)

        total = 0
        for year in years:
            total += self.build_year(year, force=force)

        print(f"\n  [build] Total: {total} daily tensors built")

        # Compute normalization statistics
        self.compute_norm_stats()

    # ------------------------------------------------------------------
    #  Compute normalization statistics (training set only)
    # ------------------------------------------------------------------
    def compute_norm_stats(self, train_years: Optional[list] = None):
        """
        Compute per-channel mean and std from TRAINING years only.

        Uses Welford's online algorithm to avoid loading all data into memory.
        Saves to data/processed/norm_stats.npz.
        """
        train_years = train_years or TRAIN_YEARS
        print(f"\n  [norm] Computing normalization stats from years: {train_years}")

        # Welford's online algorithm for mean and variance
        n_input = np.zeros(NCHAN, dtype=np.float64)
        mean_input = np.zeros(NCHAN, dtype=np.float64)
        m2_input = np.zeros(NCHAN, dtype=np.float64)

        n_target = np.zeros(NDEPTH, dtype=np.float64)
        mean_target = np.zeros(NDEPTH, dtype=np.float64)
        m2_target = np.zeros(NDEPTH, dtype=np.float64)

        n_salt = np.zeros(NDEPTH, dtype=np.float64)
        mean_salt = np.zeros(NDEPTH, dtype=np.float64)
        m2_salt = np.zeros(NDEPTH, dtype=np.float64)

        count = 0
        for year in train_years:
            year_dir = self.output_dir / str(year)
            if not year_dir.exists():
                continue

            for npz_path in sorted(year_dir.glob("*.npz")):
                data = np.load(npz_path)
                inp = data["input"]     # [H, W, 7]
                tgt = data["target"]    # [H, W, 15]
                tgt_s = data.get("target_salt", None)

                for c in range(NCHAN):
                    vals = inp[:, :, c].ravel()
                    valid = vals[~np.isnan(vals)]
                    for v in valid:
                        n_input[c] += 1
                        delta = v - mean_input[c]
                        mean_input[c] += delta / n_input[c]
                        delta2 = v - mean_input[c]
                        m2_input[c] += delta * delta2

                for d in range(NDEPTH):
                    vals = tgt[:, :, d].ravel()
                    valid = vals[~np.isnan(vals)]
                    for v in valid:
                        n_target[d] += 1
                        delta = v - mean_target[d]
                        mean_target[d] += delta / n_target[d]
                        delta2 = v - mean_target[d]
                        m2_target[d] += delta * delta2

                if tgt_s is not None:
                    salt = tgt_s if isinstance(tgt_s, np.ndarray) else np.array(tgt_s)
                    for d in range(NDEPTH):
                        vals = salt[:, :, d].ravel()
                        valid = vals[~np.isnan(vals)]
                        for v in valid:
                            n_salt[d] += 1
                            delta = v - mean_salt[d]
                            mean_salt[d] += delta / n_salt[d]
                            delta2 = v - mean_salt[d]
                            m2_salt[d] += delta * delta2

                count += 1

        if count == 0:
            print("  [norm] No training data found! Skipping normalization.")
            return

        # Compute std
        std_input = np.sqrt(m2_input / np.maximum(n_input - 1, 1)).astype(np.float32)
        std_target = np.sqrt(m2_target / np.maximum(n_target - 1, 1)).astype(np.float32)
        std_salt = np.sqrt(m2_salt / np.maximum(n_salt - 1, 1)).astype(np.float32)

        # Prevent division by zero
        std_input[std_input < 1e-6] = 1.0
        std_target[std_target < 1e-6] = 1.0
        std_salt[std_salt < 1e-6] = 1.0

        out_path = DATA_PROCESSED / "norm_stats.npz"
        np.savez(
            out_path,
            input_mean=mean_input.astype(np.float32),
            input_std=std_input,
            target_mean=mean_target.astype(np.float32),
            target_std=std_target,
            salt_mean=mean_salt.astype(np.float32),
            salt_std=std_salt,
            channels=np.array(INPUT_CHANNELS),
            depths=np.array(DEPTH_LEVELS),
            train_years=np.array(train_years),
            n_days=count,
        )
        print(f"  [norm] Saved stats from {count} days to {out_path.name}")
        print(f"  [norm] Input means:  {mean_input.astype(np.float32)}")
        print(f"  [norm] Input stds:   {std_input}")
        print(f"  [norm] Target T mean (surface): {mean_target[0]:.2f} degC")


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build training tensors from CMEMS data")
    parser.add_argument("--years", type=str, default="2015-2023")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--norm-only", action="store_true",
                        help="Only compute normalization stats")
    args = parser.parse_args()

    if "-" in args.years:
        s, e = args.years.split("-")
        years = range(int(s), int(e) + 1)
    else:
        years = range(int(args.years), int(args.years) + 1)

    builder = TrainingDataBuilder()

    if args.norm_only:
        builder.compute_norm_stats()
    else:
        builder.build_all(years, force=args.force)
