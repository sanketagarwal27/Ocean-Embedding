"""
Phase 0 — Step 3: Land/Ocean mask and coastal confidence mask.

Outputs (saved to data/masks/):
    ocean_mask.npy       — bool array [NLAT, NLON], True = ocean
    coast_mask.npy       — float [NLAT, NLON], distance to nearest coast in km
    coast_confidence.npy — float [NLAT, NLON], 0=low-conf(coastal), 1=high-conf(open ocean)

Visualizations saved to outputs/plots/:
    ocean_mask.png
    coastal_confidence.png

Method:
    Land/ocean mask is derived directly from the SST or SSH dataset
    (ocean cells are where the native product has valid (non-NaN) data).
    Distance-to-coast is computed as the geodesic distance from each ocean
    cell to the nearest land cell using scipy's KD-tree.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

from ocean_pipeline.config import (
    EXISTING_NC_FILES, DATA_MASKS, OUTPUTS_PLOTS,
    LATS, LONS, NLAT, NLON, COAST_DISTANCE_KM,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, GRID_RES
)
from ocean_pipeline.preprocessing.regrid import regrid_to_target


def build_ocean_mask(source: str = "ssh") -> np.ndarray:
    """
    Build boolean ocean mask from an existing dataset.
    Ocean = True, Land = False.
    """
    path = EXISTING_NC_FILES[source]
    if not path.exists():
        # Fall back to SSH, then SST
        for alt in ["ssh", "sst", "sss"]:
            if EXISTING_NC_FILES[alt].exists():
                path = EXISTING_NC_FILES[alt]
                source = alt
                break
        else:
            raise FileNotFoundError("No raw NetCDF available to derive ocean mask")

    print(f"  Building ocean mask from: {path.name}")
    ds = xr.open_dataset(path, engine="netcdf4")

    # Find lat/lon dim names
    lat_dim = next(c for c in ds.coords if c.lower() in ("lat", "latitude"))
    lon_dim = next(c for c in ds.coords if c.lower() in ("lon", "longitude"))

    # Pick first time slice and first (or only) variable
    var = list(ds.data_vars)[0]
    da = ds[var]

    # Take first time step if time dimension exists
    if "time" in da.dims:
        da = da.isel(time=0)
    # Take first depth if present
    for ddim in ("depth", "level", "lev", "deptht"):
        if ddim in da.dims:
            da = da.isel({ddim: 0})
            break

    # Squeeze extra dims
    da = da.squeeze()

    # Regrid to target
    da_tgt = regrid_to_target(da, LATS, LONS, lat_dim=lat_dim, lon_dim=lon_dim)
    ocean_mask = ~np.isnan(da_tgt.values)

    print(f"  Ocean mask: {ocean_mask.sum()} ocean cells / {NLAT * NLON} total")
    ds.close()
    return ocean_mask.astype(bool)


def compute_distance_to_coast(ocean_mask: np.ndarray) -> np.ndarray:
    """
    For each ocean cell, compute geodesic distance (km) to the nearest land cell.
    Uses KD-tree on lat/lon converted to 3-D Cartesian (on unit sphere).

    Returns float array [NLAT, NLON], NaN for land cells.
    """
    lon2d, lat2d = np.meshgrid(LONS, LATS)

    # Convert to radians then Cartesian (on unit sphere, multiply by R later)
    R_EARTH_KM = 6371.0
    lat_r = np.radians(lat2d)
    lon_r = np.radians(lon2d)
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    coords_3d = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

    # Land cells = ~ocean_mask
    land_idx = (~ocean_mask).ravel()
    land_coords = coords_3d[land_idx]

    if land_coords.shape[0] == 0:
        print("  WARNING: No land cells found — distance mask will be all-ocean")
        return np.full((NLAT, NLON), 1e6, dtype=np.float32)

    tree = cKDTree(land_coords)

    # Query only ocean cells (for efficiency)
    ocean_idx = ocean_mask.ravel()
    ocean_coords = coords_3d[ocean_idx]

    dists_chord, _ = tree.query(ocean_coords, k=1, workers=-1)
    # chord distance on unit sphere → arc angle → km
    dists_chord = np.clip(dists_chord, 0, 2)
    dists_arc_km = R_EARTH_KM * 2 * np.arcsin(dists_chord / 2)

    dist_map = np.full(NLAT * NLON, np.nan, dtype=np.float32)
    dist_map[ocean_idx] = dists_arc_km.astype(np.float32)
    return dist_map.reshape(NLAT, NLON)


def build_confidence_mask(dist_km: np.ndarray,
                          threshold_km: float = COAST_DISTANCE_KM) -> np.ndarray:
    """
    Build smooth confidence mask (0 → 1):
        0 at coast,  1 at open ocean.
    Uses sigmoid transition centred at threshold_km.
    """
    conf = np.where(
        np.isnan(dist_km),
        np.nan,
        1.0 / (1.0 + np.exp(-(dist_km - threshold_km) / (threshold_km * 0.3)))
    )
    return conf.astype(np.float32)


def visualize_masks(ocean_mask, dist_km, confidence, save_dir: Path):
    """Save diagnostic PNG plots of both masks."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("  matplotlib not yet installed — skipping visualization")
        return

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        HAS_CARTOPY = True
    except ImportError:
        HAS_CARTOPY = False

    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Plot 1: Ocean mask ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5),
                           subplot_kw={"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {})
    if HAS_CARTOPY:
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#d4c9a8", zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
        ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)

    im = ax.pcolormesh(LONS, LATS, ocean_mask.astype(float),
                       cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Ocean (1) / Land (0)")
    ax.set_title(f"Land/Ocean Mask — North Indian Ocean ({GRID_RES}° grid)")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    plt.tight_layout()
    plt.savefig(save_dir / "ocean_mask.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_dir / 'ocean_mask.png'}")

    # ── Plot 2: Coastal confidence ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                             subplot_kw={"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {})

    for ax, data, title, cmap, label in zip(
        axes,
        [dist_km, confidence],
        ["Distance to Coast (km)", "Coastal Confidence Mask"],
        ["hot_r", "RdYlGn"],
        ["km", "confidence (0=coastal, 1=open-ocean)"]
    ):
        if HAS_CARTOPY:
            ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
            ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)

        masked = np.ma.masked_invalid(data)
        im = ax.pcolormesh(LONS, LATS, masked, cmap=cmap)
        plt.colorbar(im, ax=ax, label=label)
        ax.set_title(title)

    plt.suptitle(f"Coastal Masks — North Indian Ocean ({COAST_DISTANCE_KM} km threshold)")
    plt.tight_layout()
    plt.savefig(save_dir / "coastal_confidence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_dir / 'coastal_confidence.png'}")


def main():
    print("\n[masks] Building land/ocean mask...")
    ocean_mask = build_ocean_mask(source="ssh")

    print("[masks] Computing distance to coast...")
    dist_km = compute_distance_to_coast(ocean_mask)

    print("[masks] Building confidence mask...")
    confidence = build_confidence_mask(dist_km)

    # Save
    DATA_MASKS.mkdir(parents=True, exist_ok=True)
    np.save(DATA_MASKS / "ocean_mask.npy", ocean_mask)
    np.save(DATA_MASKS / "dist_to_coast_km.npy", dist_km)
    np.save(DATA_MASKS / "coast_confidence.npy", confidence)
    print(f"[masks] Saved to {DATA_MASKS}")

    # Stats
    ocean_count = ocean_mask.sum()
    low_conf = (confidence < 0.5).sum() if not np.all(np.isnan(confidence)) else 0
    print(f"[masks] Ocean cells: {ocean_count} / {NLAT * NLON}")
    print(f"[masks] Low-confidence (coastal) cells: {low_conf}")

    print("[masks] Generating diagnostic plots...")
    visualize_masks(ocean_mask, dist_km, confidence, OUTPUTS_PLOTS)
    print("[masks] Done.")


if __name__ == "__main__":
    main()
