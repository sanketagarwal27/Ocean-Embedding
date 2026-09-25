import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np

from ocean_pipeline.config import (
    DATA_PROCESSED, OUTPUTS_CKPT, DEPTH_LEVELS,
    LATS, LONS, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
)
from ocean_pipeline.models.baseline import BaselineCNN

app = FastAPI(
    title="Ocean Embedding API", 
    description="API to serve predicted depth vs temperature profiles for the North Indian Ocean."
)

# Allow CORS for easy frontend integration (React, Vue, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables so we only load the model once when the server starts
MODEL = None
IN_MEAN = None
IN_STD = None
DEVICE = None

def load_resources():
    global MODEL, IN_MEAN, IN_STD, DEVICE
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Ocean Embedding model on {DEVICE}...")
    
    ckpt_path = OUTPUTS_CKPT / "phase1_best.pth"
    if not ckpt_path.exists():
        raise RuntimeError(f"Checkpoint not found at {ckpt_path}. Please train the model first.")
        
    MODEL = BaselineCNN(in_channels=7, latent_dim=64).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    MODEL.load_state_dict(ckpt["model_state_dict"])
    MODEL.eval()
    
    norm_path = DATA_PROCESSED / "norm_stats.npz"
    if norm_path.exists():
        stats = np.load(norm_path)
        IN_MEAN = torch.tensor(stats["input_mean"], dtype=torch.float32).view(-1, 1, 1).to(DEVICE)
        IN_STD = torch.tensor(stats["input_std"], dtype=torch.float32).view(-1, 1, 1).to(DEVICE)
    else:
        IN_MEAN = torch.zeros((7, 1, 1)).to(DEVICE)
        IN_STD = torch.ones((7, 1, 1)).to(DEVICE)
    print("Resources loaded successfully. API is ready.")

@app.on_event("startup")
async def startup_event():
    load_resources()

def find_nearest_idx(array, value):
    return (np.abs(array - value)).argmin()

@app.get("/api/predict_profile")
async def predict_profile(lat: float, lon: float, date: str = "2015-06-15"):
    """
    Returns the predicted depth vs temperature profile in clean JSON format.
    """
    if not (LAT_MIN <= lat <= LAT_MAX) or not (LON_MIN <= lon <= LON_MAX):
        raise HTTPException(
            status_code=400, 
            detail=f"Coordinates out of bounds. Valid NIO domain: Lat({LAT_MIN} to {LAT_MAX}), Lon({LON_MIN} to {LON_MAX})"
        )
        
    year = date.split("-")[0]
    npz_path = DATA_PROCESSED / "training" / year / f"{date}.npz"
    
    if not npz_path.exists():
        raise HTTPException(status_code=404, detail=f"No input satellite data found for date {date}")
        
    # Load daily satellite inputs
    data = np.load(npz_path)
    inp = data["input"].astype(np.float32)
    x = torch.tensor(inp).permute(2, 0, 1).to(DEVICE)
    
    # Normalize inputs
    x = torch.where(torch.isnan(x), IN_MEAN.broadcast_to(x.shape), x)
    x = (x - IN_MEAN) / IN_STD
    x = x.unsqueeze(0) # [1, 7, H, W]
    
    # Predict full grid
    with torch.no_grad():
        preds = MODEL(x, DEPTH_LEVELS) # [1, 15, H, W]
        preds = preds.squeeze(0).cpu().numpy() # [15, H, W]
        
    # Extract specific point
    lat_idx = find_nearest_idx(LATS, lat)
    lon_idx = find_nearest_idx(LONS, lon)
    pred_profile = preds[:, lat_idx, lon_idx]
    
    # Build clean JSON response (No GLORYS values)
    response_data = {
        "metadata": {
            "requested_lat": lat,
            "requested_lon": lon,
            "actual_grid_lat": float(LATS[lat_idx]),
            "actual_grid_lon": float(LONS[lon_idx]),
            "date": date
        },
        "profile": []
    }
    
    for i, depth in enumerate(DEPTH_LEVELS):
        val = pred_profile[i]
        response_data["profile"].append({
            "depth_m": float(depth),
            "temperature_c": round(float(val), 3) if not np.isnan(val) else None
        })
        
    return response_data
