import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import torch
import numpy as np
import xarray as xr

from ocean_pipeline.config import (
    DATA_PROCESSED, OUTPUTS_CKPT, DEPTH_LEVELS,
    LATS, LONS, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
)
from ocean_pipeline.models.baseline import BaselineCNN

def find_nearest_idx(array, value):
    return (np.abs(array - value)).argmin()

def main():
    parser = argparse.ArgumentParser(description="Run inference and extract (lat, lon) profile")
    parser.add_argument("--date", type=str, default="2015-01-01", help="Date to infer (YYYY-MM-DD)")
    parser.add_argument("--lat", type=float, default=15.0, help="Latitude (5 to 30)")
    parser.add_argument("--lon", type=float, default=70.0, help="Longitude (45 to 105)")
    parser.add_argument("--ckpt", type=str, default="phase1_best.pth", help="Checkpoint name")
    parser.add_argument("--out", type=str, default="pred_output.nc", help="Output NetCDF filename")
    args = parser.parse_args()

    # Setup directories
    year = args.date.split("-")[0]
    npz_path = DATA_PROCESSED / "training" / year / f"{args.date}.npz"
    norm_path = DATA_PROCESSED / "norm_stats.npz"
    ckpt_path = OUTPUTS_CKPT / args.ckpt
    
    if not npz_path.exists():
        print(f"[Error] Input data for {args.date} not found at {npz_path}")
        return
        
    if not ckpt_path.exists():
        print(f"[Error] Model checkpoint not found at {ckpt_path}")
        return

    # Check bounds
    if not (LAT_MIN <= args.lat <= LAT_MAX) or not (LON_MIN <= args.lon <= LON_MAX):
        print(f"[Warning] Requested (lat, lon) is outside the domain bounds.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference] Using device: {device}")

    # 1. Load Model
    model = BaselineCNN(in_channels=7, latent_dim=64).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 2. Load and Normalize Data
    data = np.load(npz_path)
    inp = data["input"].astype(np.float32)  # [H, W, C=7]
    target = data["target"].astype(np.float32) # [H, W, D=15]
    
    x = torch.tensor(inp).permute(2, 0, 1) # [7, H, W]
    
    if norm_path.exists():
        stats = np.load(norm_path)
        in_mean = torch.tensor(stats["input_mean"], dtype=torch.float32).view(-1, 1, 1)
        in_std = torch.tensor(stats["input_std"], dtype=torch.float32).view(-1, 1, 1)
    else:
        in_mean = torch.zeros((7, 1, 1))
        in_std = torch.ones((7, 1, 1))
        
    # Handle NaNs in input
    x = torch.where(torch.isnan(x), in_mean.broadcast_to(x.shape), x)
    x = (x - in_mean) / in_std
    x = x.unsqueeze(0).to(device) # [1, 7, H, W]

    # 3. Run Inference
    with torch.no_grad():
        preds = model(x, DEPTH_LEVELS) # [1, 15, H, W]
        preds = preds.squeeze(0).cpu().numpy() # [15, H, W]

    # 4. Extract Point Profile
    lat_idx = find_nearest_idx(LATS, args.lat)
    lon_idx = find_nearest_idx(LONS, args.lon)
    
    actual_lat = LATS[lat_idx]
    actual_lon = LONS[lon_idx]
    
    pred_profile = preds[:, lat_idx, lon_idx]
    true_profile = target[lat_idx, lon_idx, :] # target is [H, W, D]
    
    print("\n" + "="*50)
    print(f" Profile for {args.date} at (Lat: {actual_lat:.2f}, Lon: {actual_lon:.2f})")
    print("="*50)
    print(f"{'Depth (m)':<10} | {'Predicted (°C)':<15} | {'GLORYS True (°C)':<15}")
    print("-" * 50)
    
    for d_idx, depth in enumerate(DEPTH_LEVELS):
        pred_val = pred_profile[d_idx]
        true_val = true_profile[d_idx]
        t_str = f"{true_val:.3f}" if not np.isnan(true_val) else "NaN (Land/Bottom)"
        print(f"{depth:<10} | {pred_val:<15.3f} | {t_str:<15}")
        
    # 5. Export full grid to NetCDF
    out_dir = Path("outputs") / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out
    
    ds = xr.Dataset(
        data_vars={
            "temperature_pred": (("depth", "latitude", "longitude"), preds),
            "temperature_true": (("latitude", "longitude", "depth"), target)
        },
        coords={
            "depth": DEPTH_LEVELS,
            "latitude": LATS,
            "longitude": LONS,
            "time": [np.datetime64(args.date)]
        }
    )
    # Transpose true to match pred format
    ds["temperature_true"] = ds["temperature_true"].transpose("depth", "latitude", "longitude")
    
    ds.to_netcdf(out_path)
    print(f"\n[Export] Full 3D prediction grid saved to {out_path}")

if __name__ == "__main__":
    main()
