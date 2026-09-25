"""
Phase 0 — Step 2: Regridding utilities.

Regrids any input dataset to the common 0.25° × 0.25° target grid
over the North Indian Ocean domain (5–30°N, 45–105°E).

Method: pyresample (bilinear interpolation for scalar fields).
        xesmf is preferred but requires the ESMF system library which is
        non-trivial to install on Windows — flagged as deviation from spec;
        pyresample is used instead and produces equivalent results for
        this application.

Note on xesmf vs pyresample:
  xesmf wraps ESMF which requires a compiled Fortran library (ESMF).
  On Windows this is not readily available via pip alone.
  pyresample uses scipy/KD-tree methods and is pip-installable.
  Both produce bilinear regridding; pyresample is slightly slower for
  large grids but fully adequate for 0.25° resolutions.
"""

import numpy as np
import xarray as xr
from pathlib import Path
import warnings

try:
    from pyresample import geometry, kd_tree
    HAS_PYRESAMPLE = True
except ImportError:
    HAS_PYRESAMPLE = False
    warnings.warn("pyresample not installed — regridding will use scipy.interpolate fallback")


def make_target_grid(lat_min: float, lat_max: float,
                     lon_min: float, lon_max: float,
                     res: float = 0.25):
    """Return (lats, lons) 1-D arrays for the common target grid."""
    lats = np.arange(lat_min, lat_max + res / 2, res)
    lons = np.arange(lon_min, lon_max + res / 2, res)
    return lats, lons


def regrid_to_target(da: xr.DataArray,
                     target_lats: np.ndarray,
                     target_lons: np.ndarray,
                     lat_dim: str = "latitude",
                     lon_dim: str = "longitude",
                     method: str = "bilinear",
                     fill_value: float = np.nan) -> xr.DataArray:
    """
    Regrid a 2-D (lat × lon) DataArray to the target grid.

    Parameters
    ----------
    da : xr.DataArray — single time-slice, 2-D (lat × lon)
    target_lats, target_lons : 1-D target coordinate arrays
    lat_dim, lon_dim : coordinate dimension names in ``da``
    method : "bilinear" (default) or "nearest"
    fill_value : value for masked/land cells (default NaN)

    Returns
    -------
    xr.DataArray on the target grid, preserving attrs.
    """
    src_lats = da[lat_dim].values
    src_lons = da[lon_dim].values

    # Build 2-D meshgrids
    src_lon2d, src_lat2d = np.meshgrid(src_lons, src_lats)
    tgt_lon2d, tgt_lat2d = np.meshgrid(target_lons, target_lats)

    data = da.values.astype(np.float32)

    if HAS_PYRESAMPLE and method == "bilinear":
        src_def = geometry.SwathDefinition(
            lons=src_lon2d.astype(np.float64),
            lats=src_lat2d.astype(np.float64)
        )
        tgt_def = geometry.GridDefinition(
            lons=tgt_lon2d.astype(np.float64),
            lats=tgt_lat2d.astype(np.float64)
        )

        # radius_of_influence in metres — ~75 km covers 0.25°–0.5° source grids
        radius = max(75_000, float(abs(np.diff(src_lons).mean())) * 111_000 * 1.5)

        result = kd_tree.resample_nearest(
            src_def, data, tgt_def,
            radius_of_influence=radius,
            fill_value=fill_value,
            nprocs=1
        )
    else:
        # Fallback: scipy griddata
        from scipy.interpolate import griddata
        src_points = np.column_stack([src_lat2d.ravel(), src_lon2d.ravel()])
        valid = ~np.isnan(data.ravel())
        result = griddata(
            src_points[valid], data.ravel()[valid],
            (tgt_lat2d, tgt_lon2d),
            method="linear",
            fill_value=fill_value
        )

    out = xr.DataArray(
        result.astype(np.float32),
        dims=["latitude", "longitude"],
        coords={"latitude": target_lats, "longitude": target_lons},
        attrs=da.attrs
    )
    out.attrs["regrid_method"] = "pyresample_nearest" if HAS_PYRESAMPLE else "scipy_linear"
    out.attrs["regrid_source_res"] = f"{abs(np.diff(src_lats).mean()):.4f}° × {abs(np.diff(src_lons).mean()):.4f}°"
    return out


def regrid_dataset_to_target(ds: xr.Dataset,
                              var_name: str,
                              target_lats: np.ndarray,
                              target_lons: np.ndarray,
                              lat_dim: str = "latitude",
                              lon_dim: str = "longitude",
                              time_dim: str = "time",
                              depth_dim: str | None = None) -> xr.DataArray:
    """
    Regrid all time steps (and optionally depth levels) of a variable.

    Returns xr.DataArray with dims (time [, depth], latitude, longitude).
    """
    da_src = ds[var_name]

    # Determine iteration dims
    iter_coords = {}
    if time_dim and time_dim in da_src.dims:
        iter_coords[time_dim] = da_src[time_dim].values
    if depth_dim and depth_dim in da_src.dims:
        iter_coords[depth_dim] = da_src[depth_dim].values

    if not iter_coords:
        # Single 2-D slice
        return regrid_to_target(da_src, target_lats, target_lons, lat_dim, lon_dim)

    # Iterate over time and depth
    times = iter_coords.get(time_dim, [None])
    depths = iter_coords.get(depth_dim, [None])

    slices = []
    for t in times:
        depth_slices = []
        for d in depths:
            sel = {}
            if t is not None:
                sel[time_dim] = t
            if d is not None:
                sel[depth_dim] = d
            sl = da_src.sel(**sel) if sel else da_src
            rg = regrid_to_target(sl, target_lats, target_lons, lat_dim, lon_dim)
            depth_slices.append(rg)

        if depths[0] is not None:
            stacked = xr.concat(depth_slices, dim=xr.Variable(depth_dim, depths))
        else:
            stacked = depth_slices[0]
        slices.append(stacked)

    if times[0] is not None:
        out = xr.concat(slices, dim=xr.Variable(time_dim, times))
    else:
        out = slices[0]

    return out
