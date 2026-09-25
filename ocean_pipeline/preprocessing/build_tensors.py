"""
Phase 0 - Step 6: Tensor stacking pipeline.

Reads one day's surface fields and stacks into:
    input tensor  -- [H=NLAT, W=NLON, C=7] float32
    target tensor -- [H=NLAT, W=NLON, D=15] float32  (temperature at depths)

Channels (in fixed order):
    0: sst          (SST in degC)
    1: sss          (SSS in PSU)
    2: ssh          (SSH/ADT in m)
    3: u_current    (surface U, m/s -- OSCAR if available, ugos fallback)
    4: v_current    (surface V, m/s -- OSCAR if available, vgos fallback)
    5: u_wind       (10m wind U, m/s -- CCMP if available, else NaN)
    6: v_wind       (10m wind V, m/s -- CCMP if available, else NaN)

Missing channels are filled with NaN and logged in metadata.
OSCAR fallback: DUACS geostrophic currents (ugos, vgos) are used when
OSCAR is not yet downloaded -- this is a documented temporary proxy.

Saves cached .npz files to data/processed/sample_days/.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import xarray as xr
import warnings
from datetime import date
from typing import Optional

from ocean_pipeline.config import (
    LATS, LONS, NLAT, NLON, NCHAN, NDEPTH,
    INPUT_CHANNELS, DEPTH_LEVELS, DATA_PROCESSED, DATA_MASKS,
    OUTPUTS_PLOTS, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
)
from ocean_pipeline.ingestion.ingesters import (
    SSTIngester, SSHIngester, SSSIngester,
    OSCARIngester, CCMPIngester, GLORYSIngester
)

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def _get_slice(da: xr.DataArray, target_date: date) -> Optional[np.ndarray]:
    """
    Extract a single-day 2D field [NLAT, NLON] from a DataArray.
    Tries exact match, then accepts the only available slice if unique.
    Returns None if no match found.
    """
    if len(da.time) == 0:
        return None

    import pandas as pd
    target_ts = pd.Timestamp(target_date)

    # Try exact match
    try:
        sl = da.sel(time=target_ts, method="nearest", tolerance=pd.Timedelta("1D"))
        return sl.values
    except Exception:
        pass

    # If only one time step in file, use it regardless of date
    if len(da.time) == 1:
        return da.isel(time=0).values

    return None


def build_daily_tensor(target_date: date,
                       ocean_mask: Optional[np.ndarray] = None,
                       save: bool = True) -> tuple:
    """
    Build input [H, W, 7] and target [H, W, 15] tensors for a single date.

    When a channel file exists but doesn't contain that exact date,
    the closest available date is used (with a warning) so that
    pipeline plumbing can be tested on single-snapshot sample files.

    Returns (input_tensor, target_tensor, channel_status_dict)
    """
    ds = target_date.strftime("%Y-%m-%d")

    input_tensor  = np.full((NLAT, NLON, NCHAN), np.nan, dtype=np.float32)
    target_tensor = np.full((NLAT, NLON, NDEPTH), np.nan, dtype=np.float32)
    channel_status = {}

    # ---- SST ----------------------------------------------------------------
    try:
        sst_ing = SSTIngester()
        avail = sst_ing.get_available_dates()
        if avail:
            d_str = avail[0].strftime("%Y-%m-%d")
            da = sst_ing.get_daily_field(date_range=(d_str, d_str))
            if len(da.time) > 0:
                input_tensor[:, :, INPUT_CHANNELS.index("sst")] = da.isel(time=0).values
                channel_status["sst"] = f"OK({d_str})"
            else:
                channel_status["sst"] = "EMPTY"
        else:
            channel_status["sst"] = "NO_FILE"
    except Exception as e:
        channel_status["sst"] = f"FAIL:{e}"

    # ---- SSH ----------------------------------------------------------------
    ssh_da = None
    try:
        ssh_ing = SSHIngester()
        avail = ssh_ing.get_available_dates()
        if avail:
            d_str = avail[0].strftime("%Y-%m-%d")
            da = ssh_ing.get_daily_field(date_range=(d_str, d_str))
            if len(da.time) > 0:
                ssh_da = da
                input_tensor[:, :, INPUT_CHANNELS.index("ssh")] = da.isel(time=0).values
                channel_status["ssh"] = f"OK({d_str})"
            else:
                channel_status["ssh"] = "EMPTY"
        else:
            channel_status["ssh"] = "NO_FILE"
    except Exception as e:
        channel_status["ssh"] = f"FAIL:{e}"

    # ---- SSS ----------------------------------------------------------------
    try:
        sss_ing = SSSIngester()
        avail = sss_ing.get_available_dates()
        if avail:
            d_str = avail[0].strftime("%Y-%m-%d")
            da = sss_ing.get_daily_field(date_range=(d_str, d_str))
            if len(da.time) > 0:
                input_tensor[:, :, INPUT_CHANNELS.index("sss")] = da.isel(time=0).values
                channel_status["sss"] = f"OK({d_str})"
            else:
                channel_status["sss"] = "EMPTY"
        else:
            channel_status["sss"] = "NO_FILE"
    except Exception as e:
        channel_status["sss"] = f"FAIL:{e}"

    # ---- Currents: OSCAR (preferred) -> DUACS geostrophic (fallback) --------
    oscar_loaded = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u_curr, v_curr = OSCARIngester().get_daily_field()
        # OSCARIngester returns NaN stubs
        channel_status["u_current"] = "NaN_STUB"
        channel_status["v_current"] = "NaN_STUB"
    except Exception as e:
        channel_status["u_current"] = f"FAIL:{e}"
        channel_status["v_current"] = f"FAIL:{e}"

    # OSCAR not available -- try DUACS geostrophic fallback
    if not oscar_loaded:
        try:
            ssh_ing2 = SSHIngester()
            avail = ssh_ing2.get_available_dates()
            if avail:
                d_str = avail[0].strftime("%Y-%m-%d")
                u_geo, v_geo = ssh_ing2.get_geostrophic_currents(date_range=(d_str, d_str))
                if u_geo is not None and len(u_geo.time) > 0:
                    input_tensor[:, :, INPUT_CHANNELS.index("u_current")] = u_geo.isel(time=0).values
                    input_tensor[:, :, INPUT_CHANNELS.index("v_current")] = v_geo.isel(time=0).values
                    channel_status["u_current"] = f"GEO_FALLBACK({d_str})"
                    channel_status["v_current"] = f"GEO_FALLBACK({d_str})"
        except Exception as e:
            pass  # already NaN from OSCAR stub

    # ---- Winds: CCMP (stub) -------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u_wind_stub, v_wind_stub = CCMPIngester().get_daily_field()
    input_tensor[:, :, INPUT_CHANNELS.index("u_wind")] = u_wind_stub.isel(time=0).values
    input_tensor[:, :, INPUT_CHANNELS.index("v_wind")] = v_wind_stub.isel(time=0).values
    channel_status["u_wind"] = "NaN_STUB"
    channel_status["v_wind"] = "NaN_STUB"

    # ---- GLORYS target (stub) -----------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        temp_da, salt_da = GLORYSIngester().get_daily_field()
    # target_tensor stays NaN until GLORYS is downloaded
    channel_status["glorys"] = "NaN_STUB"

    # ---- Apply land mask ----------------------------------------------------
    if ocean_mask is not None:
        land = ~ocean_mask
        input_tensor[land, :] = np.nan

    # ---- Save to disk -------------------------------------------------------
    if save:
        out_dir = DATA_PROCESSED / "sample_days"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ds}.npz"
        np.savez_compressed(
            out_path,
            input=input_tensor,
            target=target_tensor,
            lats=LATS,
            lons=LONS,
            date=np.array([ds]),
            channels=np.array(INPUT_CHANNELS),
            depths=np.array(DEPTH_LEVELS),
            channel_status=np.array(list(channel_status.items()))
        )
        print(f"  [tensor] Saved {out_path.name}")

    return input_tensor, target_tensor, channel_status


def load_daily_tensor(target_date: date) -> tuple:
    """Load a previously saved tensor pair from .npz cache."""
    ds = target_date.strftime("%Y-%m-%d")
    path = DATA_PROCESSED / "sample_days" / f"{ds}.npz"
    if not path.exists():
        raise FileNotFoundError(f"No cached tensor for {ds}. Run build_daily_tensor() first.")
    data = np.load(path, allow_pickle=True)
    return data["input"], data["target"]


def visualize_sample_day(tensor_input: np.ndarray,
                         tensor_target: np.ndarray,
                         target_date: date,
                         ocean_mask: Optional[np.ndarray] = None,
                         save_dir: Path = OUTPUTS_PLOTS):
    """
    Plot each of the 7 input channels + one sample GLORYS temperature profile.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available -- skipping visualization")
        return

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        HAS_CARTOPY = True
    except ImportError:
        HAS_CARTOPY = False

    ds_str = target_date.strftime("%Y-%m-%d")
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- 7 input channels ---------------------------------------------------
    proj_kw = {"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {}
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), subplot_kw=proj_kw)
    fig.suptitle(f"Input Channels -- {ds_str}", fontsize=14)

    cmaps = {
        "sst":       ("RdYlBu_r", "degC"),
        "sss":       ("YlOrRd",   "PSU"),
        "ssh":       ("seismic",  "m"),
        "u_current": ("RdBu_r",   "m/s"),
        "v_current": ("RdBu_r",   "m/s"),
        "u_wind":    ("PuOr",     "m/s"),
        "v_wind":    ("PuOr",     "m/s"),
    }

    axes_flat = axes.flatten()
    for i, ch in enumerate(INPUT_CHANNELS):
        ax = axes_flat[i]
        if HAS_CARTOPY:
            ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor="#d4c9a8", zorder=2)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)

        data = tensor_input[:, :, i]
        masked = np.ma.masked_invalid(data)
        cmap, unit = cmaps.get(ch, ("viridis", ""))

        if ch in ("u_current", "v_current", "u_wind", "v_wind"):
            valid = data[~np.isnan(data)]
            vabs = float(np.percentile(np.abs(valid), 95)) if len(valid) > 0 else 1.0
            im = ax.pcolormesh(LONS, LATS, masked, cmap=cmap,
                               vmin=-vabs, vmax=vabs)
        else:
            im = ax.pcolormesh(LONS, LATS, masked, cmap=cmap)

        plt.colorbar(im, ax=ax, label=unit, shrink=0.7)
        ax.set_title(ch, fontsize=10)

    axes_flat[-1].set_visible(False)
    plt.tight_layout()
    save_path = save_dir / f"sample_day_inputs_{ds_str}.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [viz] Saved: {save_path.name}")

    # ---- Temperature profile (centre of domain) -----------------------------
    if not np.all(np.isnan(tensor_target)):
        from ocean_pipeline.config import DEPTH_LEVELS
        fig, ax = plt.subplots(figsize=(5, 8))
        lat_idx = np.argmin(np.abs(LATS - 15.0))
        lon_idx = np.argmin(np.abs(LONS - 75.0))
        profile = tensor_target[lat_idx, lon_idx, :]
        if not np.all(np.isnan(profile)):
            ax.plot(profile, DEPTH_LEVELS, "b-o", markersize=4)
            ax.invert_yaxis()
            ax.set_ylabel("Depth (m)")
            ax.set_xlabel("Temperature (degC)")
            ax.set_title(f"GLORYS T-profile  15N,75E  {ds_str}")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            ppath = save_dir / f"sample_profile_{ds_str}.png"
            plt.savefig(ppath, dpi=120, bbox_inches="tight")
            plt.close()
            print(f"  [viz] Saved: {ppath.name}")
    else:
        print(f"  [viz] GLORYS target is NaN -- profile plot skipped (not yet downloaded)")


# ---- CLI entry point --------------------------------------------------------
if __name__ == "__main__":
    mask_path = DATA_MASKS / "ocean_mask.npy"
    ocean_mask = np.load(mask_path) if mask_path.exists() else None

    # Use SSH available date as a demo date
    from ocean_pipeline.ingestion.ingesters import SSHIngester
    avail = SSHIngester().get_available_dates()
    test_date = avail[0] if avail else date(2026, 1, 16)

    print(f"\n[stack] Building tensor for {test_date}...")
    inp, tgt, status = build_daily_tensor(test_date, ocean_mask=ocean_mask, save=True)
    print(f"  input shape: {inp.shape}  target shape: {tgt.shape}")
    for k, v in status.items():
        print(f"  {k:15s}: {v}")

    visualize_sample_day(inp, tgt, test_date, ocean_mask=ocean_mask)
    print("\n[stack] Done.")
