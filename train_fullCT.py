import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys
import numpy as np

sys.path.append('/host/d/GitHub_folder/CT_registration_diffusion')
from create_fullCT_dataloader import Dataset_FullCT

# ==================== Model ====================
class SpatialTransformer(nn.Module):
    """Warp image using deformation field"""
    def __init__(self, size):
        super().__init__()
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]
        
        return F.grid_sample(src, new_locs, align_corners=True, mode='bilinear')

class UNet3D(nn.Module):
    """3D U-Net for registration"""
    def __init__(self, in_channels=2, out_channels=3):
        super().__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 16)
        self.enc2 = self.conv_block(16, 32)
        self.enc3 = self.conv_block(32, 64)
        self.enc4 = self.conv_block(64, 128)
        
        # Decoder
        self.up3 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.dec3 = self.conv_block(128, 64)
        self.up2 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec2 = self.conv_block(64, 32)
        self.up1 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.dec1 = self.conv_block(32, 16)
        
        # Output
        self.flow = nn.Conv3d(16, out_channels, 1)
        self.flow.weight = nn.Parameter(torch.zeros_like(self.flow.weight))
        self.flow.bias = nn.Parameter(torch.zeros_like(self.flow.bias))
        
        self.pool = nn.MaxPool3d(2)
        
    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, moving, fixed):
        x = torch.cat([moving, fixed], dim=1)
        
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        
        # Decoder
        x = self.up3(x4)
        x = self.dec3(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))
        
        flow = self.flow(x)
        return flow

# ==================== Losses ====================
class NCC:
    """Normalized Cross Correlation"""
    def __init__(self, win=9):
        self.win = win

    def __call__(self, I, J):
        ndims = len(I.shape) - 2
        win_size = self.win ** ndims
        
        # Compute filters
        sum_filt = torch.ones(1, 1, *([self.win] * ndims)).to(I.device)
        pad_no = self.win // 2
        stride = tuple([1] * ndims)
        padding = tuple([pad_no] * ndims)
        
        # Compute local sums
        I_sum = F.conv3d(I, sum_filt, stride=stride, padding=padding)
        J_sum = F.conv3d(J, sum_filt, stride=stride, padding=padding)
        I2_sum = F.conv3d(I * I, sum_filt, stride=stride, padding=padding)
        J2_sum = F.conv3d(J * J, sum_filt, stride=stride, padding=padding)
        IJ_sum = F.conv3d(I * J, sum_filt, stride=stride, padding=padding)
        
        # Compute cross correlation
        I_mean = I_sum / win_size
        J_mean = J_sum / win_size
        
        cross = IJ_sum - J_mean * I_sum - I_mean * J_sum + I_mean * J_mean * win_size
        I_var = I2_sum - 2 * I_mean * I_sum + I_mean * I_mean * win_size
        J_var = J2_sum - 2 * J_mean * J_sum + J_mean * J_mean * win_size
        
        cc = cross * cross / (I_var * J_var + 1e-5)
        return -torch.mean(cc)

class Grad:
    """Gradient regularization"""
    def __init__(self, penalty='l2'):
        self.penalty = penalty

    def __call__(self, flow):
        dy = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
        dx = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])
        dz = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])
        
        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz
        
        return (torch.mean(dx) + torch.mean(dy) + torch.mean(dz)) / 3.0

# ==================== Training ====================
def train():
    # Config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    num_epochs = 100
    lr = 1e-4
    checkpoint_dir = "/host/d/GitHub_folder/CT_registration_diffusion/checkpoints_fullCT"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Data
    marker_root = '/host/d/data/4DCT/DIR_LAB_original_0203/markers_final'
    cases = sorted([os.path.join(marker_root, d) for d in os.listdir(marker_root) if d.startswith('Case')])
    
    train_cases = cases[:8]
    val_cases = cases[8:]
    
    train_dataset = Dataset_FullCT(train_cases, target_shape=(128, 128, 128))
    val_dataset = Dataset_FullCT(val_cases, target_shape=(128, 128, 128))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Model
    model = UNet3D(in_channels=2, out_channels=3).to(device)
    spatial_transform = SpatialTransformer(size=(128, 128, 128)).to(device)
    
    # Loss
    ncc_loss = NCC(win=9)
    grad_loss = Grad(penalty='l2')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_ncc = 0
        train_grad = 0
        
        for moving, fixed in train_loader:
            moving, fixed = moving.to(device), fixed.to(device)
            
            # Forward
            flow = model(moving, fixed)
            warped = spatial_transform(moving, flow)
            
            # Loss
            loss_ncc = ncc_loss(warped, fixed)
            loss_grad = grad_loss(flow)
            loss = loss_ncc + 0.01 * loss_grad
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_ncc += loss_ncc.item()
            train_grad += loss_grad.item()
        
        train_loss /= len(train_loader)
        train_ncc /= len(train_loader)
        train_grad /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_ncc = 0
        
        with torch.no_grad():
            for moving, fixed in val_loader:
                moving, fixed = moving.to(device), fixed.to(device)
                flow = model(moving, fixed)
                warped = spatial_transform(moving, flow)
                loss_ncc = ncc_loss(warped, fixed)
                loss_grad = grad_loss(flow)
                loss = loss_ncc + 0.01 * loss_grad
                val_loss += loss.item()
                val_ncc += loss_ncc.item()
        
        val_loss /= len(val_loader)
        val_ncc /= len(val_loader)
        
        print(f"Epoch {epoch+1:03d} | Train: {train_loss:.4f} (NCC:{train_ncc:.4f}, Grad:{train_grad:.4f}) | Val: {val_loss:.4f} (NCC:{val_ncc:.4f})")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch+1}.pth"))
            print(f"  ✅ Saved checkpoint")

if __name__ == "__main__":
    train()

