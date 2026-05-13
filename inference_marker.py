import torch
import torch.nn as nn
import os
import sys
import numpy as np
from tqdm import tqdm

# Add paths
sys.path.append('/host/d/GitHub_folder/CT_registration_diffusion')
sys.path.append('/host/d/GitHub_folder/CT_registration_diffusion/model')

from Generator_marker import Dataset_MarkerCT
from model import Unet

# Load checkpoint
checkpoint_path = "/host/d/GitHub_folder/CT_registration_diffusion/checkpoints_marker/checkpoint_epoch100.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate model with same architecture
model = Unet(
    problem_dimension="3D",
    input_channels=2,
    out_channels=3,
    initial_dim=8,
    dim_mults=(2, 4, 8),
    groups=2,
    attn_dim_head=16,
    attn_heads=2,
    full_attn_paths=(None, None, None),
    full_attn_bottleneck=False,
    act="ReLU"
).to(device)

# Load weights
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
print(f"Train Loss: {checkpoint['train_loss']:.4f}, Val Loss: {checkpoint['val_loss']:.4f}")

# Load test data (validation cases)
marker_root = '/host/d/data/4DCT/DIR_LAB_original_0203/markers_final'
cases = sorted([os.path.join(marker_root, d) for d in os.listdir(marker_root) if d.startswith('Case')])
test_cases = cases[8:]  # Use validation cases for testing

test_ds = Dataset_MarkerCT(test_cases, cutoff_range=None, normalize_factor=None)

# Create output directory
output_dir = "/host/d/GitHub_folder/CT_registration_diffusion/results_marker"
os.makedirs(output_dir, exist_ok=True)

# Run inference
print(f"\nRunning inference on {len(test_ds)} test samples...")
with torch.no_grad():
    for idx in tqdm(range(len(test_ds))):
        moving, fixed = test_ds[idx]
        
        # Convert to tensor if needed
        if isinstance(moving, np.ndarray):
            moving = torch.from_numpy(moving).float()
        if isinstance(fixed, np.ndarray):
            fixed = torch.from_numpy(fixed).float()
        
        moving = moving.unsqueeze(0).to(device)
        fixed = fixed.unsqueeze(0).to(device)
        
        # Generate deformation field
        x = torch.cat([moving, fixed], dim=1)
        pred_flow = model(x)
        
        # Save results
        pred_flow_np = pred_flow.cpu().numpy()[0]  # Shape: (3, D, H, W)
        moving_np = moving.cpu().numpy()[0, 0]
        fixed_np = fixed.cpu().numpy()[0, 0]
        
        np.save(os.path.join(output_dir, f"flow_{idx:03d}.npy"), pred_flow_np)
        np.save(os.path.join(output_dir, f"moving_{idx:03d}.npy"), moving_np)
        np.save(os.path.join(output_dir, f"fixed_{idx:03d}.npy"), fixed_np)
        
        if idx < 3:  # Print first 3 samples
            print(f"\nSample {idx}:")
            print(f"  Moving shape: {moving_np.shape}")
            print(f"  Flow shape: {pred_flow_np.shape}")
            print(f"  Flow range: [{pred_flow_np.min():.3f}, {pred_flow_np.max():.3f}]")

print(f"\n✅ Results saved to: {output_dir}")
print(f"Total files: {len(os.listdir(output_dir))}")