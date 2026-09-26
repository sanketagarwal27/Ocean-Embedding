import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import numpy as np
import torch
from datetime import date
import pandas as pd

from ocean_pipeline.config import (
    DATA_ARGO, DATA_PROCESSED, OUTPUTS_CKPT, DEPTH_LEVELS,
    LATS, LONS, TEST_YEARS, TRAIN_YEARS, VAL_YEARS
)
from ocean_pipeline.models.baseline import BaselineCNN
from ocean_pipeline.training.dataset import OceanDataset
from torch.utils.data import DataLoader

def load_model(ckpt_path, device):
    model = BaselineCNN(in_channels=7, latent_dim=64).to(device)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def load_norm_stats(norm_path):
    stats = np.load(norm_path)
    in_mean = torch.tensor(stats["input_mean"], dtype=torch.float32).view(-1, 1, 1)
    in_std = torch.tensor(stats["input_std"], dtype=torch.float32).view(-1, 1, 1)
    return in_mean, in_std

def load_input_tensor(date_str, data_dir, in_mean, in_std, device):
    """Load a specific day's input tensor for inference."""
    year = date_str[:4]
    npz_path = data_dir / year / f"{date_str}.npz"
    
    if not npz_path.exists():
        return None
        
    data = np.load(npz_path)
    inp = data["input"].astype(np.float32)
    x = torch.tensor(inp).permute(2, 0, 1)
    
    # Fill NaNs with mean (0 after norm)
    x = torch.where(torch.isnan(x), in_mean.broadcast_to(x.shape), x)
    x = (x - in_mean) / in_std
    
    return x.unsqueeze(0).to(device) # (1, C, H, W)

def evaluate_argo(model, device, eval_years=TEST_YEARS):
    data_dir = DATA_PROCESSED / "training"
    norm_path = DATA_PROCESSED / "norm_stats.npz"
    
    in_mean, in_std = load_norm_stats(norm_path)
    
    errors = [] # Store (pred - true) for each depth
    
    for year in eval_years:
        argo_path = DATA_ARGO / f"argo_colocated_{year}.npz"
        if not argo_path.exists():
            print(f"[Eval] Skipping year {year}: ARGO file not found.")
            continue
            
        argo_data = np.load(argo_path, allow_pickle=True)
        dates = argo_data["date"]
        grid_lats = argo_data["grid_lat_idx"]
        grid_lons = argo_data["grid_lon_idx"]
        temp_true = argo_data["temp_15lev"] # (N, 15)
        
        print(f"[Eval] Evaluating {len(dates)} profiles for year {year}...")
        
        # We process unique dates to avoid running the model multiple times for the same day
        unique_dates = np.unique(dates)
        
        for d_str in unique_dates:
            # Find all profiles for this day
            idx = np.where(dates == d_str)[0]
            
            # Run inference for this day
            x = load_input_tensor(d_str, data_dir, in_mean, in_std, device)
            if x is None:
                continue
                
            with torch.no_grad():
                preds = model(x, DEPTH_LEVELS) # (1, 15, H, W)
                preds = preds.squeeze(0).cpu().numpy() # (15, H, W)
                
            # Compare with ARGO profiles
            for i in idx:
                lat_idx = grid_lats[i]
                lon_idx = grid_lons[i]
                
                pred_profile = preds[:, lat_idx, lon_idx]
                true_profile = temp_true[i]
                
                # Calculate error where true data exists
                valid = ~np.isnan(true_profile)
                if np.any(valid):
                    diff = pred_profile[valid] - true_profile[valid]
                    
                    # Store results [depth_idx, error]
                    for d_idx, err in zip(np.where(valid)[0], diff):
                        errors.append({"depth_idx": d_idx, "error": err})
                        
    if len(errors) == 0:
        print("[Eval] No matching dates found between ARGO and inputs.")
        return
        
    df = pd.DataFrame(errors)
    
    # Calculate metrics per depth
    print("\n--- Evaluation Results vs ARGO ---")
    print(f"{'Depth (m)':<10} | {'RMSE (°C)':<10} | {'MAE (°C)':<10} | {'Bias (°C)':<10} | {'N Profiles'}")
    print("-" * 65)
    
    overall_mse = 0
    overall_mae = 0
    total_points = 0
    
    for d_idx, depth in enumerate(DEPTH_LEVELS):
        depth_errs = df[df["depth_idx"] == d_idx]["error"].values
        if len(depth_errs) > 0:
            rmse = np.sqrt(np.mean(depth_errs**2))
            mae = np.mean(np.abs(depth_errs))
            bias = np.mean(depth_errs)
            count = len(depth_errs)
            
            overall_mse += np.sum(depth_errs**2)
            overall_mae += np.sum(np.abs(depth_errs))
            total_points += count
            
            print(f"{depth:<10} | {rmse:<10.3f} | {mae:<10.3f} | {bias:<10.3f} | {count}")
            
    overall_rmse = np.sqrt(overall_mse / total_points)
    overall_mae_final = overall_mae / total_points
    print("-" * 65)
    print(f"Overall RMSE: {overall_rmse:.3f} °C")
    print(f"Overall MAE:  {overall_mae_final:.3f} °C")
    print(f"Total Points: {total_points}")

def evaluate_glorys(model, device):
    """Evaluates the model against the GLORYS dataset (test set)."""
    print("\n[Eval] Initializing GLORYS evaluation...")
    data_dir = DATA_PROCESSED / "training"
    norm_stats = DATA_PROCESSED / "norm_stats.npz"
    
    # Instantiate directly to avoid empty train_loader bug in get_dataloaders
    test_dataset = OceanDataset(data_dir=data_dir, norm_stats_path=norm_stats, years=TEST_YEARS)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    if len(test_loader) == 0:
        print("[Eval] No testing data found.")
        return

    print(f"[Eval] Evaluating over {len(test_loader)} batches...")
    
    # Track errors per depth level
    # sum_sq_err[depth_idx], sum_abs_err[depth_idx], count[depth_idx]
    depth_stats = {d_idx: {"sq_err": 0.0, "abs_err": 0.0, "bias": 0.0, "count": 0} for d_idx in range(len(DEPTH_LEVELS))}
    
    with torch.no_grad():
        for x, y, mask, dates in test_loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            
            preds = model(x, DEPTH_LEVELS) # (B, D, H, W)
            
            # mask out land and invalid data
            mask_d = mask.unsqueeze(1).expand_as(preds)
            valid_targets = ~torch.isnan(y)
            final_mask = mask_d & valid_targets
            
            # Compute stats per depth
            for d_idx in range(len(DEPTH_LEVELS)):
                m = final_mask[:, d_idx, :, :]
                if m.sum() > 0:
                    p = preds[:, d_idx, :, :][m]
                    t = y[:, d_idx, :, :][m]
                    
                    diff = (p - t).cpu().numpy()
                    
                    depth_stats[d_idx]["sq_err"] += np.sum(diff**2)
                    depth_stats[d_idx]["abs_err"] += np.sum(np.abs(diff))
                    depth_stats[d_idx]["bias"] += np.sum(diff)
                    depth_stats[d_idx]["count"] += len(diff)

    print("\n--- Evaluation Results vs GLORYS Reanalysis ---")
    print(f"{'Depth (m)':<10} | {'RMSE (°C)':<10} | {'MAE (°C)':<10} | {'Bias (°C)':<10} | {'N Points'}")
    print("-" * 65)
    
    overall_mse = 0
    overall_mae = 0
    overall_bias = 0
    total_points = 0
    
    for d_idx, depth in enumerate(DEPTH_LEVELS):
        stats = depth_stats[d_idx]
        if stats["count"] > 0:
            rmse = np.sqrt(stats["sq_err"] / stats["count"])
            mae = stats["abs_err"] / stats["count"]
            bias = stats["bias"] / stats["count"]
            count = stats["count"]
            
            overall_mse += stats["sq_err"]
            overall_mae += stats["abs_err"]
            overall_bias += stats["bias"]
            total_points += count
            
            print(f"{depth:<10} | {rmse:<10.3f} | {mae:<10.3f} | {bias:<10.3f} | {count}")
            
    if total_points > 0:
        overall_rmse = np.sqrt(overall_mse / total_points)
        overall_mae_final = overall_mae / total_points
        overall_bias_final = overall_bias / total_points
        print("-" * 65)
        print(f"Overall RMSE: {overall_rmse:.3f} °C")
        print(f"Overall MAE:  {overall_mae_final:.3f} °C")
        print(f"Overall Bias: {overall_bias_final:.3f} °C")
        print(f"Total Points: {total_points}")
    else:
        print("No valid points evaluated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Phase 1 against ARGO")
    parser.add_argument("--ckpt", type=str, default="phase1_best.pth")
    parser.add_argument("--argo", action="store_true", help="Evaluate against ARGO floats")
    parser.add_argument("--glorys", action="store_true", help="Evaluate against GLORYS test set")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = OUTPUTS_CKPT / args.ckpt
    
    try:
        model = load_model(ckpt_path, device)
        
        # Default to GLORYS if nothing specified
        if not args.argo and not args.glorys:
            args.glorys = True
            
        if args.argo:
            evaluate_argo(model, device)
            
        if args.glorys:
            evaluate_glorys(model, device)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Error] {e}")
