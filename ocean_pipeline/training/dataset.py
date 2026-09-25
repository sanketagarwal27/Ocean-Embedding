import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class OceanDataset(Dataset):
    """
    PyTorch Dataset for Ocean Embedding.
    Reads pre-built daily .npz tensors from data/processed/training/.
    
    Expected input shape: [NCHAN, NLAT, NLON]
    Expected target shape: [NDEPTH, NLAT, NLON]
    """
    def __init__(self, data_dir, norm_stats_path, years, split="train", temporal_window=1):
        self.data_dir = Path(data_dir)
        self.years = years
        self.split = split
        self.temporal_window = temporal_window
        
        # Discover all available files for requested years
        self.files = []
        for year in self.years:
            year_dir = self.data_dir / str(year)
            if year_dir.exists():
                self.files.extend(sorted(year_dir.glob("*.npz")))
                
        if len(self.files) == 0:
            print(f"[{split}] Warning: No data files found in {self.data_dir} for years {self.years}")
            
        # Load normalization stats
        if Path(norm_stats_path).exists():
            stats = np.load(norm_stats_path)
            self.in_mean = torch.tensor(stats["input_mean"], dtype=torch.float32).view(-1, 1, 1)
            self.in_std = torch.tensor(stats["input_std"], dtype=torch.float32).view(-1, 1, 1)
        else:
            print(f"[{split}] Warning: Norm stats not found at {norm_stats_path}. Using zero mean, unit variance.")
            self.in_mean = torch.zeros((7, 1, 1), dtype=torch.float32) # NCHAN=7
            self.in_std = torch.ones((7, 1, 1), dtype=torch.float32)

    def __len__(self):
        # With temporal window, we can only start at index (temporal_window - 1)
        # For Phase 1a (single day), temporal_window = 1, so len is len(files)
        return max(0, len(self.files) - self.temporal_window + 1)

    def __getitem__(self, idx):
        """
        Returns:
            x: (Channels, H, W) for single day, or (Window*Channels, H, W) if window > 1
            y: (Depth, H, W) -> GLORYS temperature target
            mask: (H, W) -> valid ocean cells
            date_str: ISO date string
        """
        # Load the target day (the end of the temporal window)
        target_idx = idx + self.temporal_window - 1
        data = np.load(self.files[target_idx])
        
        # Load target (temperature)
        # target_tensor: [H, W, D=15] -> convert to [D, H, W]
        target = data["target"].astype(np.float32)
        y = torch.tensor(target).permute(2, 0, 1)
        date_str = str(data["date"][0])
        
        # Ocean mask: true where target has valid data at ANY depth
        mask = (~torch.isnan(y)).any(dim=0)
        
        if self.temporal_window == 1:
            # Single day input: [H, W, C=7] -> convert to [C, H, W]
            inp = data["input"].astype(np.float32)
            x = torch.tensor(inp).permute(2, 0, 1)
            
            # Mask NaNs in input with zeros for processing (after norm, zero is mean)
            x = torch.where(torch.isnan(x), self.in_mean.broadcast_to(x.shape), x)
            
            # Normalize
            x = (x - self.in_mean) / self.in_std
            
        else:
            # Temporal stacking (Phase 1b)
            inputs = []
            for i in range(idx, idx + self.temporal_window):
                d = np.load(self.files[i])
                inp = d["input"].astype(np.float32)
                xi = torch.tensor(inp).permute(2, 0, 1)
                xi = torch.where(torch.isnan(xi), self.in_mean.broadcast_to(xi.shape), xi)
                xi = (xi - self.in_mean) / self.in_std
                inputs.append(xi)
                
            x = torch.cat(inputs, dim=0) # [Window*C, H, W]
            
        return x, y, mask, date_str

def get_dataloaders(data_dir, norm_stats_path, train_years, val_years, test_years, batch_size=4, num_workers=0, temporal_window=1):
    train_dataset = OceanDataset(data_dir, norm_stats_path, train_years, "train", temporal_window)
    val_dataset = OceanDataset(data_dir, norm_stats_path, val_years, "val", temporal_window)
    test_dataset = OceanDataset(data_dir, norm_stats_path, test_years, "test", temporal_window)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
