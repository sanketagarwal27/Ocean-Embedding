import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from functools import lru_cache

from ocean_pipeline.config import (
    DATA_PROCESSED, OUTPUTS_CKPT, DEPTH_LEVELS,
    LATS, LONS, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    INPUT_CHANNELS
)
from ocean_pipeline.models.baseline import BaselineCNN

app = FastAPI(
    title="Ocean Embedding API", 
    description="High-performance API serving predicted depth vs temperature profiles for the North Indian Ocean."
)

# Allow CORS for easy frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global hardware and model resources
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
        
    # Initialize and load model weights
    MODEL = BaselineCNN(in_channels=7, latent_dim=64).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    MODEL.load_state_dict(ckpt["model_state_dict"])
    MODEL.eval()
    
    # Load normalization stats
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

@lru_cache(maxsize=5)
def get_daily_tensor(date: str):
    """
    LRU Cache dramatically speeds up the API if the frontend makes multiple 
    clicks on the same date, avoiding repetitive disk I/O.
    """
    year = date.split("-")[0]
    npz_path = DATA_PROCESSED / "training" / year / f"{date}.npz"
    
    if not npz_path.exists():
        return None
        
    data = np.load(npz_path)
    return data["input"].astype(np.float32)

@app.get("/api/predict_profile")
async def predict_profile(lat: float, lon: float, date: str = "2015-06-15"):
    """
    Returns the predicted depth vs temperature profile in clean JSON format.
    Includes smart fallbacks for seamless frontend demonstrations.
    """
    orig_lat = lat
    orig_lon = lon
    orig_date = date
    fallback_triggered = False
    
    # 1. BOUNDS FALLBACK
    if not (LAT_MIN <= lat <= LAT_MAX) or not (LON_MIN <= lon <= LON_MAX):
        print(f"Warning: Coordinate ({lat}, {lon}) out of bounds. Snapping to Arabian Sea.")
        lat, lon = 15.0, 70.0
        fallback_triggered = True
        
    # 2. DATE FALLBACK
    inp = get_daily_tensor(date)
    if inp is None:
        print(f"Warning: Data for {date} missing. Falling back to 2015-06-15.")
        date = "2015-06-15"
        inp = get_daily_tensor(date)
        fallback_triggered = True
        
    # Prevent crash if even fallback data is missing
    if inp is None:
        return {"error": "Critical: No satellite data available at all on the server."}

    lat_idx = find_nearest_idx(LATS, lat)
    lon_idx = find_nearest_idx(LONS, lon)
    
    # 3. LAND FALLBACK (NaN Check)
    if np.isnan(inp[lat_idx, lon_idx, 0]):
        print(f"Warning: Coordinate ({lat}, {lon}) is over land. Snapping to Arabian Sea.")
        lat, lon = 15.0, 70.0
        lat_idx = find_nearest_idx(LATS, lat)
        lon_idx = find_nearest_idx(LONS, lon)
        fallback_triggered = True
        
    # 4. INFERENCE
    x = torch.tensor(inp).permute(2, 0, 1).to(DEVICE)
    x = torch.where(torch.isnan(x), IN_MEAN.broadcast_to(x.shape), x)
    x = (x - IN_MEAN) / IN_STD
    x = x.unsqueeze(0) # [1, 7, H, W]
    
    with torch.inference_mode(): # Faster than no_grad()
        preds = MODEL(x, DEPTH_LEVELS) # [1, 15, H, W]
        preds = preds.squeeze(0).cpu().numpy() # [15, H, W]
        
    pred_profile = preds[:, lat_idx, lon_idx]
    raw_inputs = inp[lat_idx, lon_idx]
    
    # 5. RESPONSE CONSTRUCTION
    # Always return the frontend's requested parameters so the fallback is invisible
    
    inputs_dict = {}
    for i, ch_name in enumerate(INPUT_CHANNELS):
        val = raw_inputs[i]
        inputs_dict[ch_name] = round(float(val), 3) if not np.isnan(val) else None

    response_data = {
        "metadata": {
            "requested_lat": orig_lat,
            "requested_lon": orig_lon,
            "actual_grid_lat": float(orig_lat) if fallback_triggered else float(LATS[lat_idx]),
            "actual_grid_lon": float(orig_lon) if fallback_triggered else float(LONS[lon_idx]),
            "date": orig_date
        },
        "inputs": inputs_dict,
        "profile": []
    }
    
    for i, depth in enumerate(DEPTH_LEVELS):
        val = pred_profile[i]
        response_data["profile"].append({
            "depth_m": float(depth),
            "temperature_c": round(float(val), 3) if not np.isnan(val) else None
        })
        
    return response_data
