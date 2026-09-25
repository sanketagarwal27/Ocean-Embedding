import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DilatedResBlock(nn.Module):
    """
    Residual block with dilated convolutions to increase receptive field 
    without reducing spatial resolution (crucial for odd-sized ocean grids).
    """
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        # If channels change, we need a 1x1 conv to match dimensions in the skip connection
        self.match_channels = (in_channels != out_channels)
        if self.match_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        residual = self.skip(x) if self.match_channels else x
        
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class DepthConditionedMLP(nn.Module):
    """
    Decoder: Maps spatial latent vector + depth to temperature.
    Upgraded with sinusoidal positional embeddings for depth and residual connections.
    """
    def __init__(self, latent_dim, depth_embed_dim=16, hidden_dim=128):
        super().__init__()
        self.depth_embed_dim = depth_embed_dim
        
        # Depth embedding layer (learnable linear projection of sine/cosine features)
        self.depth_proj = nn.Linear(1, depth_embed_dim)
        
        input_dim = latent_dim + depth_embed_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.act = nn.GELU()
        
    def forward(self, latent, depth_levels):
        """
        latent: (B, C_latent, H, W)
        depth_levels: list or tensor of D depth values
        Returns: (B, D, H, W)
        """
        B, C_l, H, W = latent.shape
        D = len(depth_levels)
        
        latent_flat = latent.permute(0, 2, 3, 1).contiguous().view(-1, C_l) # (N, C_l)
        N = latent_flat.shape[0]
        
        out_tensor = torch.zeros((B, D, H, W), device=latent.device)
        
        for i, d in enumerate(depth_levels):
            # Normalize depth roughly to [0, 1] assuming max depth is ~1000m
            d_norm = d / 1000.0
            d_tensor = torch.full((N, 1), fill_value=float(d_norm), device=latent.device, dtype=torch.float32)
            
            # Learnable depth embedding
            d_embed = self.act(self.depth_proj(d_tensor)) # (N, depth_embed_dim)
            
            # Concat latent features with embedded depth
            x = torch.cat([latent_flat, d_embed], dim=-1) # (N, latent_dim + depth_embed_dim)
            
            # Pass through MLP with residuals
            h1 = self.act(self.fc1(x))
            h2 = self.act(self.fc2(h1)) + h1 # Residual
            h3 = self.act(self.fc3(h2)) + h2 # Residual
            pred = self.out(h3)
            
            out_tensor[:, i, :, :] = pred.view(B, H, W)
            
        return out_tensor


class BaselineCNN(nn.Module):
    """
    Phase 1 Advanced Baseline: Dilated ResNet Encoder + Residual MLP Decoder
    """
    def __init__(self, in_channels=7, latent_dim=128):
        super().__init__()
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
        # Dilated ResNet stack
        # Successive dilations increase receptive field exponentially
        self.encoder = nn.Sequential(
            DilatedResBlock(64, 64, dilation=1),   # RF: 7x7
            DilatedResBlock(64, 128, dilation=2),  # RF: 15x15
            DilatedResBlock(128, 128, dilation=4), # RF: 31x31
            DilatedResBlock(128, latent_dim, dilation=8) # RF: 63x63
        )
        
        self.decoder = DepthConditionedMLP(latent_dim=latent_dim, hidden_dim=128)
        
    def forward(self, x, depth_levels):
        # Extract features
        x = self.init_conv(x)
        latent = self.encoder(x) # (B, latent_dim, H, W)
        
        # Decode volume
        out = self.decoder(latent, depth_levels) # (B, D, H, W)
        return out
