import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from ocean_pipeline.config import (
    DATA_PROCESSED, OUTPUTS_CKPT, LOGS_DIR, DEPTH_LEVELS, 
    TRAIN_YEARS, VAL_YEARS, TEST_YEARS
)
from ocean_pipeline.training.dataset import get_dataloaders
from ocean_pipeline.models.baseline import BaselineCNN

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    valid_batches = 0
    
    for batch_idx, (x, y, mask, dates) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device) # (B, H, W)
        
        optimizer.zero_grad()
        
        # Forward
        preds = model(x, DEPTH_LEVELS) # (B, D, H, W)
        
        # We only compute loss where we have valid GLORYS target data (ocean mask)
        # Expand mask to match D
        mask_d = mask.unsqueeze(1).expand_as(preds)
        
        # Filter NaNs in target (e.g., if some depths hit the seafloor)
        valid_targets = ~torch.isnan(y)
        final_mask = mask_d & valid_targets
        
        if final_mask.sum() == 0:
            continue
            
        loss = criterion(preds[final_mask], y[final_mask])
        
        # Backward
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        valid_batches += 1
        
    return running_loss / max(1, valid_batches)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    valid_batches = 0
    
    with torch.no_grad():
        for x, y, mask, dates in dataloader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            
            preds = model(x, DEPTH_LEVELS)
            
            mask_d = mask.unsqueeze(1).expand_as(preds)
            valid_targets = ~torch.isnan(y)
            final_mask = mask_d & valid_targets
            
            if final_mask.sum() == 0:
                continue
                
            loss = criterion(preds[final_mask], y[final_mask])
            running_loss += loss.item()
            valid_batches += 1
            
    return running_loss / max(1, valid_batches)

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Baseline CNN Training")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase1] Using device: {device}")
    
    # 1. Setup Dataloaders
    data_dir = DATA_PROCESSED / "training"
    norm_stats = DATA_PROCESSED / "norm_stats.npz"
    
    print("[Phase1] Loading datasets...")
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=data_dir,
        norm_stats_path=norm_stats,
        train_years=TRAIN_YEARS,
        val_years=VAL_YEARS,
        test_years=TEST_YEARS,
        batch_size=args.batch_size
    )
    
    if len(train_loader) == 0:
        print("[Phase1] ERROR: No training data found. Did you run run_fetch_and_build.py?")
        return
        
    # 2. Setup Model, Optimizer, Loss
    model = BaselineCNN(in_channels=7, latent_dim=64).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # 3. Setup Logging
    writer = SummaryWriter(log_dir=str(LOGS_DIR / "tensorboard" / "phase1_baseline"))
    best_val_loss = float("inf")
    
    # 4. Training Loop
    print(f"[Phase1] Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        t0 = time.time()
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)
        
        t1 = time.time()
        
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        
        print(f"Epoch {epoch+1:02d}/{args.epochs} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | Time: {t1-t0:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = OUTPUTS_CKPT / "phase1_best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss
            }, ckpt_path)
            print(f"  -> Saved new best checkpoint: {ckpt_path.name}")
            
    writer.close()
    print("[Phase1] Training complete.")

if __name__ == "__main__":
    main()
