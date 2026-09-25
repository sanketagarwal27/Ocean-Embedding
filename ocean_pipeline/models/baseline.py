import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthConditionedMLP(nn.Module):
    def __init__(self, in_features, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        layers = []
        # Input to MLP is the pixel-level latent vector + depth value (1 feature)
        current_dim = in_features + 1 
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.GELU())
            current_dim = h_dim
            
        layers.append(nn.Linear(current_dim, 1)) # Output is temperature at that depth
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, latent, depth_levels):
        """
        latent: (B, C_latent, H, W)
        depth_levels: list or tensor of D depth values
        Returns: (B, D, H, W)
        """
        B, C_l, H, W = latent.shape
        D = len(depth_levels)
        
        # Reshape latent to (B, H, W, C_l) -> (B*H*W, C_l)
        latent_flat = latent.permute(0, 2, 3, 1).reshape(-1, C_l)
        N = latent_flat.shape[0] # N = B*H*W
        
        out = torch.zeros((B, D, H, W), device=latent.device)
        
        # We query the MLP for each depth level.
        # Alternatively, we could batch across depths too, but doing a loop over D=15 is small enough.
        for i, d in enumerate(depth_levels):
            # Create depth tensor: (N, 1)
            d_tensor = torch.full((N, 1), fill_value=float(d), device=latent.device, dtype=torch.float32)
            
            # Concat latent and depth: (N, C_l + 1)
            x = torch.cat([latent_flat, d_tensor], dim=-1)
            
            # Predict: (N, 1)
            pred = self.mlp(x)
            
            # Reshape back to (B, H, W) and store
            out[:, i, :, :] = pred.view(B, H, W)
            
        return out


class BaselineCNN(nn.Module):
    """
    Phase 1 Baseline Model: Simple CNN Encoder + Depth-Conditioned MLP Decoder
    """
    def __init__(self, in_channels=7, latent_dim=64):
        super().__init__()
        
        # Simple ResNet-style blocks or just a deep CNN
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            nn.Conv2d(64, latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.GELU()
        )
        
        self.decoder = DepthConditionedMLP(in_features=latent_dim)
        
    def forward(self, x, depth_levels):
        """
        x: (B, in_channels, H, W)
        depth_levels: list of depths (length D)
        """
        # Encode to latent grid
        latent = self.encoder(x) # (B, latent_dim, H, W)
        
        # Decode at each depth level
        out = self.decoder(latent, depth_levels) # (B, D, H, W)
        
        return out
