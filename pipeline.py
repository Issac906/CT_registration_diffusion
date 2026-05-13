"""  
=============================================================  
CT 图像配准  完整评估 Pipeline  
=============================================================  
整合来源:  
  inference_marker.py          → Unet + Dataset_MarkerCT  
  compare_mvf_with_markers.py  → UNet3D + Dataset_FullCT + 真实TRE  
  comprehensive_evaluation.py  → UNet3D + Dataset_FullCT + 图像质量指标  

使用方法:  
  1. 只修改最顶部的 CONFIG 字典  
  2. python pipeline.py  

新数据适配要点:  
  - 修改 data_root / nii_root / case_prefix  
  - 修改 fixed_marker / moving_marker 文件名格式  
  - 修改 split_mode 选择使用哪些 Case  
  - 根据需要选择 pipeline_mode: 'fullCT' 或 'marker'  
=============================================================  
"""  

import os  
import sys  
import logging  
import json  
import warnings  
from datetime import datetime  

import numpy as np  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from torch.utils.data import Dataset, DataLoader  

import nibabel as nib  
from scipy import ndimage  
from skimage.metrics import structural_similarity as ssim_func  

import matplotlib  
matplotlib.use('Agg')  
import matplotlib.pyplot as plt  
from model.model import Unet as TrainingUnet

warnings.filterwarnings('ignore')  


# =============================================================  
#  CONFIG — 使用新数据时只需修改这里  
# =============================================================  
CONFIG = {  

    # ── 选择运行哪条 Pipeline ─────────────────────────────────  
    # 'fullCT'  : UNet3D + Dataset_FullCT  (来自 train_fullCT.py 路线)  
    # 'marker'  : Unet   + Dataset_MarkerCT (来自 inference_marker.py 路线)  
    # 'both'    : 两条都跑  
    'pipeline_mode': 'fullCT',  

    # ── 路径 ──────────────────────────────────────────────────  
    'project_root': '/host/d/Github/CT_registration_diffusion',  

    # markers_final 目录（存放 Case1~Case10 文件夹，含 txt marker文件）  
    'marker_root': '/host/d/data/CT_image/DIR_LAB',  

    # nii.gz 目录（Dataset_FullCT 用，存放 case1~case10/case*_T00.nii.gz）  
    'nii_root': '/host/d/data/CT_image/DIR_LAB',  

    # 输出目录  
    'output_root': '/host/d/pipeline_results',  

    # ── Checkpoint 路径 ───────────────────────────────────────  
    # fullCT 路线使用  
    'ckpt_fullCT': '/host/d/projects/registration/models/trial_1/models_stage2/model-210.pt',

    # marker 路线使用  
    # 'ckpt_marker': '/host/d/GitHub_folder/CT_registration_diffusion'  
    #                '/checkpoints_marker/checkpoint_epoch100.pth',  

    # ── 数据划分 ───────────────────────────────────────────────  
    # 'test'   → 按 train_ratio 划分后的后段（原始代码: cases[8:]）  
    # 'train'  → 前段  
    # 'all'    → 全部  
    # 'index'  → 手动指定 case_indices  
    'split_mode': 'test',  
    'train_ratio': 0.8,  
    'case_indices': [8, 9],      # split_mode='index' 时生效  

    # ── 数据命名规则（换数据集时修改）────────────────────────────  
    'case_prefix': 'Case',               # Case目录前缀，如 'Case' / 'Patient'  
    # marker 文件名（相对于每个 case 目录）  
    'fixed_marker_name': 'marker/{case_name_lower}_dirLab300_T00_xyz.txt',  
    'moving_marker_name': 'marker/{case_name_lower}_dirLab300_T50_xyz.txt',  
    # nii.gz 文件名（相对于 nii_root/{case_name_lower}/）  
    'fixed_nii_name': 'resampled_image/img_0.nii.gz',  
    'moving_nii_name': 'resampled_image/img_5.nii.gz',  
    # marker CT 文件名（Dataset_MarkerCT 用）  
    'marker_ct_suffix': '_ct_with_markers_cropped.nii.gz',  

    # ── 模型参数（与训练时保持一致）──────────────────────────────  
    # UNet3D (fullCT 路线)  
    'fullCT_target_shape': (128, 128, 128),  

    # Unet (marker 路线)  
    'marker_image_size': (224, 224, 96),  
    'unet_input_channels': 2,  
    'unet_out_channels': 3,  
    'unet_initial_dim': 16,  
    'unet_dim_mults': (2, 4, 8),  
    'unet_groups': 2,  
    'unet_attn_heads': 2,  
    'unet_full_attn_paths': (None, None, None),  
    'unet_full_attn_bottleneck': None,  

    # ── 评估参数 ───────────────────────────────────────────────  
    'tre_thresholds': [1.0, 2.0, 3.0],  # voxels  
    'ssim_slice_axis': 0,               # 0=axial, 1=coronal, 2=sagittal  
    'ssim_slice_idx': 64,  
    'ncc_win': 9,  

    # ── 输出控制 ───────────────────────────────────────────────  
    'save_flow_npy': True,  
    'save_warped_npy': True,  
    'save_figures': True,  
    'save_json': True,  

    # ── 设备 ───────────────────────────────────────────────────  
    'device': 'auto',   # 'auto' | 'cpu' | 'cuda:0'  
}  


# =============================================================  
#  日志  
# =============================================================  

def setup_logger(output_dir: str) -> logging.Logger:  
    os.makedirs(output_dir, exist_ok=True)  
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')  
    log_path = os.path.join(output_dir, f'pipeline_{ts}.log')  
    fmt = '%(asctime)s  %(levelname)-8s  %(message)s'  
    logging.basicConfig(  
        level=logging.INFO,  
        format=fmt,  
        handlers=[  
            logging.FileHandler(log_path, encoding='utf-8'),  
            logging.StreamHandler(sys.stdout),  
        ]  
    )  
    return logging.getLogger('pipeline')  


# =============================================================  
#  设备  
# =============================================================  

def get_device(cfg: dict) -> torch.device:  
    if cfg['device'] == 'auto':  
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
    return torch.device(cfg['device'])  


# =============================================================  
#  数据工具  
# =============================================================  

def get_case_list(cfg: dict) -> list:  
    """返回 Case 目录路径列表（根据 split_mode）"""  
    root   = cfg['marker_root']  
    prefix = cfg['case_prefix']  

    all_cases = sorted([  
        os.path.join(root, d)  
        for d in os.listdir(root)  
        if d.startswith(prefix) and os.path.isdir(os.path.join(root, d))  
    ])  
    if not all_cases:  
        raise FileNotFoundError(  
            f"在 {root} 下未找到以 '{prefix}' 开头的目录"  
        )  

    mode = cfg['split_mode']  
    if mode == 'all':  
        return all_cases  
    elif mode == 'train':  
        n = max(1, int(len(all_cases) * cfg['train_ratio']))  
        return all_cases[:n]  
    elif mode == 'test':  
        n = max(1, int(len(all_cases) * cfg['train_ratio']))  
        return all_cases[n:]  
    elif mode == 'index':  
        return [all_cases[i] for i in cfg['case_indices'] if i < len(all_cases)]  
    else:  
        raise ValueError(f"未知的 split_mode: {mode}")  


def load_markers(case_dir: str, cfg: dict):  
    """  
    读取 T00 / T50 Marker 坐标文件  
    返回: t00_markers [N,3], t50_markers [N,3]  单位: voxels  
    """  
    case_name = os.path.basename(case_dir)  
    case_name_lower = case_name.lower()  

    t00_fname = cfg['fixed_marker_name'].format(
        case_name=case_name,
        case_name_lower=case_name_lower
    )  
    t50_fname = cfg['moving_marker_name'].format(
        case_name=case_name,
        case_name_lower=case_name_lower
    )  

    t00_path = os.path.join(case_dir, t00_fname)  
    t50_path = os.path.join(case_dir, t50_fname)  

    def read_txt(fpath):  
        if not os.path.exists(fpath):  
            raise FileNotFoundError(f"Marker文件不存在: {fpath}")  
        coords = []  
        with open(fpath, 'r') as f:  
            for line in f:  
                line = line.strip()  
                if not line or line.startswith('#'):  
                    continue  
                vals = line.split()  
                if len(vals) >= 3:  
                    coords.append([float(v) for v in vals[:3]])  
        return np.array(coords, dtype=np.float32)  

    t00 = read_txt(t00_path)  
    t50 = read_txt(t50_path)  

    if len(t00) != len(t50):  
        raise ValueError(  
            f"{case_name}: Marker数量不匹配 "  
            f"T00={len(t00)}, T50={len(t50)}"  
        )  
    return t00, t50   # [N,3], [N,3]  


# =============================================================  
#  Dataset_FullCT（直接从 create_fullCT_dataloader.py 复制，完全一致）  
# =============================================================  

class Dataset_FullCT(Dataset):  
    """  
    与 create_fullCT_dataloader.py 完全一致  
    读取 nii.gz CT 图像，clip HU[-1000,500] → normalize [0,1] → zoom to target_shape  
    """  
    def __init__(self, case_dirs, nii_root, target_shape=(128, 128, 128),  
                 normalize=True,  
                 fixed_nii_name='{case_name_lower}_T00.nii.gz',  
                 moving_nii_name='{case_name_lower}_T50.nii.gz'):  
        self.target_shape = target_shape  
        self.normalize    = normalize  
        self.pairs        = []   # (case_name, moving_path, fixed_path)  

        for case_dir in case_dirs:  
            case_name       = os.path.basename(case_dir)  
            case_name_lower = case_name.lower()  
            t00_rel = fixed_nii_name.format(
                case_name=case_name,
                case_name_lower=case_name_lower
            )
            t50_rel = moving_nii_name.format(
                case_name=case_name,
                case_name_lower=case_name_lower
            )

            candidates_t00 = [
                os.path.join(case_dir, t00_rel),
                os.path.join(nii_root, case_name, t00_rel),
                os.path.join(nii_root, case_name_lower, t00_rel),
            ]
            candidates_t50 = [
                os.path.join(case_dir, t50_rel),
                os.path.join(nii_root, case_name, t50_rel),
                os.path.join(nii_root, case_name_lower, t50_rel),
            ]

            t00_path = next((p for p in candidates_t00 if os.path.exists(p)),
                            candidates_t00[0])
            t50_path = next((p for p in candidates_t50 if os.path.exists(p)),
                            candidates_t50[0])

            if os.path.exists(t00_path) and os.path.exists(t50_path):  
                self.pairs.append((case_name, t50_path, t00_path))  
            else:  
                print(f"  ⚠  跳过 {case_name}：找不到nii.gz文件")  
                print(f"     期望 T00: {t00_path}")  
                print(f"     期望 T50: {t50_path}")  

        print(f"Dataset_FullCT: 找到 {len(self.pairs)} 对有效CT")  

    def __len__(self):  
        return len(self.pairs)  

    def load_and_preprocess(self, path):  
        img = nib.load(path).get_fdata().astype(np.float32)  
        img = np.clip(img, -1000, 500)               # HU window  
        if self.normalize:  
            img = (img + 1000) / 1500.0              # → [0, 1]  
        zoom_factors = [t / s for t, s in zip(self.target_shape, img.shape)]  
        img = ndimage.zoom(img, zoom_factors, order=1)  
        return img  

    def __getitem__(self, idx):  
        _, moving_path, fixed_path = self.pairs[idx]  
        moving = self.load_and_preprocess(moving_path)  
        fixed  = self.load_and_preprocess(fixed_path)  
        moving = torch.from_numpy(moving).float().unsqueeze(0)  # [1,D,H,W]  
        fixed  = torch.from_numpy(fixed).float().unsqueeze(0)  
        return moving, fixed  

    def get_case_name(self, idx):  
        """返回第 idx 对的 case 名称（用于保存文件）"""  
        return self.pairs[idx][0]  


# =============================================================  
#  Dataset_MarkerCT（直接从 Generator_marker.py 复制，完全一致）  
# =============================================================  

def _center_crop_or_pad_3d(vol, target_shape, pad_val=0):  
    x, y, z    = vol.shape  
    tx, ty, tz = target_shape  
    out = np.full((tx, ty, tz), pad_val, dtype=vol.dtype)  
    sx0 = max((x - tx) // 2, 0); sx1 = sx0 + min(tx, x)  
    sy0 = max((y - ty) // 2, 0); sy1 = sy0 + min(ty, y)  
    sz0 = max((z - tz) // 2, 0); sz1 = sz0 + min(tz, z)  
    dx0 = max((tx - x) // 2, 0); dx1 = dx0 + (sx1 - sx0)  
    dy0 = max((ty - y) // 2, 0); dy1 = dy0 + (sy1 - sy0)  
    dz0 = max((tz - z) // 2, 0); dz1 = dz0 + (sz1 - sz0)  
    out[dx0:dx1, dy0:dy1, dz0:dz1] = vol[sx0:sx1, sy0:sy1, sz0:sz1]  
    return out  


class Dataset_MarkerCT(Dataset):  
    """  
    与 Generator_marker.py 完全一致  
    读取 Case*_T00/T50_ct_with_markers_cropped.nii.gz  
    """  
    def __init__(self, case_dirs, image_size=(224, 224, 96),  
                 marker_ct_suffix='_ct_with_markers_cropped.nii.gz'):  
        self.case_dirs          = case_dirs  
        self.image_size         = tuple(image_size)  
        self.marker_ct_suffix   = marker_ct_suffix  

    def __len__(self):  
        return len(self.case_dirs)  

    def _find_t00_t50(self, case_dir):  
        import glob, re  
        case_name = os.path.basename(case_dir)  
        suffix    = self.marker_ct_suffix  

        t00 = os.path.join(case_dir, f"{case_name}_T00{suffix}")  
        t50 = os.path.join(case_dir, f"{case_name}_T50{suffix}")  

        # fallback: resampled  
        if not os.path.exists(t00):  
            t00 = os.path.join(  
                case_dir,  
                f"{case_name}_T00_ct_with_markers_resampled.nii.gz"  
            )  
        if not os.path.exists(t50):  
            t50 = os.path.join(  
                case_dir,  
                f"{case_name}_T50_ct_with_markers_resampled.nii.gz"  
            )  

        # fallback: glob  
        if not (os.path.exists(t00) and os.path.exists(t50)):  
            cand = sorted(glob.glob(  
                os.path.join(case_dir, f"{case_name}_T*_ct_with_markers*.nii.gz")  
            ))  
            if len(cand) < 2:  
                raise FileNotFoundError(f"找不到T00/T50文件: {case_dir}")  
            def tf(p):  
                m = re.search(r'_T(\d+)_', os.path.basename(p))  
                return int(m.group(1)) if m else 10**9  
            cand = sorted(cand, key=tf)  
            t00 = next((p for p in cand if '_T00_' in p), cand[0])  
            t50 = next((p for p in cand if '_T50_' in p), cand[-1])  

        return t00, t50  

    def __getitem__(self, idx):  
        case_dir = self.case_dirs[idx]  
        t00, t50 = self._find_t00_t50(case_dir)  
        moving = nib.load(t00).get_fdata().astype(np.float32)  
        fixed  = nib.load(t50).get_fdata().astype(np.float32)  
        if moving.shape != self.image_size:  
            moving = _center_crop_or_pad_3d(moving, self.image_size, pad_val=0)  
            fixed  = _center_crop_or_pad_3d(fixed,  self.image_size, pad_val=0)  
        moving = torch.from_numpy(moving[None]).float()   # [1,X,Y,Z]  
        fixed  = torch.from_numpy(fixed[None]).float()  
        return moving, fixed  


# =============================================================  
#  UNet3D（从 train_fullCT.py 完全复制）  
# =============================================================  

class SpatialTransformer(nn.Module):  
    """与 train_fullCT.py 完全一致"""  
    def __init__(self, size):  
        super().__init__()  
        vectors = [torch.arange(0, s) for s in size]  
        grids   = torch.meshgrid(vectors, indexing='ij')  
        grid    = torch.stack(grids).unsqueeze(0).float()  
        self.register_buffer('grid', grid)  

    def forward(self, src, flow):  
        new_locs = self.grid + flow  
        shape    = flow.shape[2:]  
        for i in range(len(shape)):  
            new_locs[:, i] = 2 * (new_locs[:, i] / (shape[i] - 1) - 0.5)  
        new_locs = new_locs.permute(0, 2, 3, 4, 1)  
        new_locs = new_locs[..., [2, 1, 0]]  
        return F.grid_sample(src, new_locs, align_corners=True, mode='bilinear')  


class UNet3D(nn.Module):  
    """与 train_fullCT.py 完全一致"""  
    def __init__(self, in_channels=2, out_channels=3):  
        super().__init__()  
        self.enc1 = self._conv_block(in_channels, 16)  
        self.enc2 = self._conv_block(16, 32)  
        self.enc3 = self._conv_block(32, 64)  
        self.enc4 = self._conv_block(64, 128)  
        self.up3  = nn.ConvTranspose3d(128, 64,  2, stride=2)  
        self.dec3 = self._conv_block(128, 64)  
        self.up2  = nn.ConvTranspose3d(64,  32,  2, stride=2)  
        self.dec2 = self._conv_block(64,  32)  
        self.up1  = nn.ConvTranspose3d(32,  16,  2, stride=2)  
        self.dec1 = self._conv_block(32,  16)  
        self.flow = nn.Conv3d(16, out_channels, 1)  
        self.flow.weight = nn.Parameter(torch.zeros_like(self.flow.weight))  
        self.flow.bias   = nn.Parameter(torch.zeros_like(self.flow.bias))  
        self.pool = nn.MaxPool3d(2)  

    @staticmethod  
    def _conv_block(in_c, out_c):  
        return nn.Sequential(  
            nn.Conv3d(in_c, out_c, 3, padding=1),  
            nn.BatchNorm3d(out_c),  
            nn.ReLU(inplace=True),  
            nn.Conv3d(out_c, out_c, 3, padding=1),  
            nn.BatchNorm3d(out_c),  
            nn.ReLU(inplace=True),  
        )  

    def forward(self, moving, fixed):  
        x  = torch.cat([moving, fixed], dim=1)  
        x1 = self.enc1(x)  
        x2 = self.enc2(self.pool(x1))  
        x3 = self.enc3(self.pool(x2))  
        x4 = self.enc4(self.pool(x3))  
        x  = self.dec3(torch.cat([self.up3(x4), x3], dim=1))  
        x  = self.dec2(torch.cat([self.up2(x),  x2], dim=1))  
        x  = self.dec1(torch.cat([self.up1(x),  x1], dim=1))  
        return self.flow(x)  


# =============================================================  
#  NCC / Grad（从 train_fullCT.py 完全复制）  
# =============================================================  

class NCC:  
    """与 train_fullCT.py 完全一致"""  
    def __init__(self, win=9):  
        self.win = win  

    def __call__(self, I, J):  
        ndims    = len(I.shape) - 2  
        win_size = self.win ** ndims  
        filt     = torch.ones(1, 1, *([self.win] * ndims)).to(I.device)  
        pad      = self.win // 2  
        stride   = tuple([1] * ndims)  
        padding  = tuple([pad]  * ndims)  

        I_sum  = F.conv3d(I,     filt, stride=stride, padding=padding)  
        J_sum  = F.conv3d(J,     filt, stride=stride, padding=padding)  
        I2_sum = F.conv3d(I * I, filt, stride=stride, padding=padding)  
        J2_sum = F.conv3d(J * J, filt, stride=stride, padding=padding)  
        IJ_sum = F.conv3d(I * J, filt, stride=stride, padding=padding)  

        I_mean = I_sum / win_size  
        J_mean = J_sum / win_size  
        cross  = IJ_sum - J_mean * I_sum - I_mean * J_sum + I_mean * J_mean * win_size  
        I_var  = I2_sum - 2 * I_mean * I_sum + I_mean * I_mean * win_size  
        J_var  = J2_sum - 2 * J_mean * J_sum + J_mean * J_mean * win_size  
        cc     = cross * cross / (I_var * J_var + 1e-5)  
        return -torch.mean(cc)  


class Grad:  
    """与 train_fullCT.py 完全一致"""  
    def __init__(self, penalty='l2'):  
        self.penalty = penalty  

    def __call__(self, flow):  
        dy = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])  
        dx = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])  
        dz = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])  
        if self.penalty == 'l2':  
            dy, dx, dz = dy * dy, dx * dx, dz * dz  
        return (torch.mean(dx) + torch.mean(dy) + torch.mean(dz)) / 3.0  


# =============================================================  
#  Unet（从 model.py 提取，去掉所有外部依赖，只保留推理所需部分）  
# =============================================================  

def _exists(x):   return x is not None  
def _default(v, d): return v if _exists(v) else (d() if callable(d) else d)  
def _cast_tuple(t, length=1): return t if isinstance(t, tuple) else (t,) * length  


class _RMSNorm3D(nn.Module):  
    def __init__(self, dim):  
        super().__init__()  
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))  

    def forward(self, x):  
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)  


class _ConvBlock3D(nn.Module):  
    def __init__(self, dim, dim_out, groups=8, act='ReLU'):  
        super().__init__()  
        self.conv = nn.Conv3d(dim, dim_out, 3, padding=1)  
        self.norm = nn.GroupNorm(groups, dim_out)  
        self.act  = nn.ReLU() if act == 'ReLU' else nn.LeakyReLU()  

    def forward(self, x):  
        return self.act(self.norm(self.conv(x)))  


class _LinearAttn3D(nn.Module):  
    def __init__(self, dim, heads=4, dim_head=32):  
        super().__init__()  
        self.scale  = dim_head ** -0.5  
        self.heads  = heads  
        hidden      = dim_head * heads  
        self.norm   = _RMSNorm3D(dim)  
        self.to_qkv = nn.Conv3d(dim, hidden * 3, 1, bias=False)  
        self.to_out = nn.Sequential(nn.Conv3d(hidden, dim, 1), _RMSNorm3D(dim))  

    def forward(self, x):  
        b, c, h, w, d = x.shape  
        x   = self.norm(x)  
        q, k, v = self.to_qkv(x).chunk(3, dim=1)  

        def reshape(t):  
            return t.reshape(b, self.heads, -1, h * w * d)  

        q, k, v = reshape(q), reshape(k), reshape(v)  
        q = q.softmax(dim=-2) * self.scale  
        k = k.softmax(dim=-1)  
        ctx = torch.einsum('b h d n, b h e n -> b h d e', k, v)  
        out = torch.einsum('b h d e, b h d n -> b h e n', ctx, q)  
        out = out.reshape(b, -1, h, w, d)  
        return self.to_out(out)  


class _ResnetBlock3D(nn.Module):  
    def __init__(self, dim, dim_out, groups=8,  
                 use_full_attention=None,  
                 attn_head=4, attn_dim_head=32, act='ReLU'):  
        super().__init__()  
        self.block1 = _ConvBlock3D(dim,     dim_out, groups=groups, act=act)  
        self.block2 = _ConvBlock3D(dim_out, dim_out, groups=groups, act=act)  
        if use_full_attention is False:  
            self.attention = _LinearAttn3D(dim_out, heads=attn_head, dim_head=attn_dim_head)  
        else:  
            self.attention = nn.Identity()  
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()  

    def forward(self, x):  
        h = self.block2(self.block1(x))  
        h = self.attention(h)  
        return h + self.res_conv(x)  


def _Upsample3D(dim, dim_out=None, upsample_factor=(2, 2, 1)):  
    return nn.Sequential(  
        nn.Upsample(scale_factor=upsample_factor, mode='nearest'),  
        nn.Conv3d(dim, _default(dim_out, dim), 3, padding=1)  
    )  



def _Downsample3D(dim, dim_out=None):
    return nn.Sequential(
        nn.MaxPool3d(kernel_size=(2, 2, 2)),
        nn.Conv3d(dim, _default(dim_out, dim), 3, padding=1)
    )


class Unet(nn.Module):
    """
    与 model/model.py 完全一致（推理专用，去掉时间步嵌入）
    """
    def __init__(
        self,
        problem_dimension='3D',
        input_channels=2,
        out_channels=3,
        initial_dim=16,
        dim_mults=(2, 4, 8),
        groups=2,
        attn_dim_head=16,
        attn_heads=2,
        full_attn_paths=(None, None, None),
        full_attn_bottleneck=None,
        act='ReLU',
    ):
        super().__init__()

        dims = [initial_dim, *[initial_dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_stages = len(in_out)
        full_attn_paths = _cast_tuple(full_attn_paths, num_stages)

        # 初始卷积
        self.init_conv = nn.Conv3d(input_channels, initial_dim, 7, padding=3)

        # Encoder
        self.downs = nn.ModuleList()
        for idx, ((dim_in, dim_out), attn_flag) in enumerate(
            zip(in_out, full_attn_paths)
        ):
            self.downs.append(nn.ModuleList([
                _ResnetBlock3D(dim_in, dim_in, groups=groups,
                               use_full_attention=attn_flag,
                               attn_head=attn_heads,
                               attn_dim_head=attn_dim_head, act=act),
                _ResnetBlock3D(dim_in, dim_in, groups=groups,
                               use_full_attention=attn_flag,
                               attn_head=attn_heads,
                               attn_dim_head=attn_dim_head, act=act),
                _Downsample3D(dim_in, dim_out),
            ]))

        # Bottleneck
        mid_dim = dims[-1]
        self.mid_block1 = _ResnetBlock3D(
            mid_dim, mid_dim, groups=groups,
            use_full_attention=full_attn_bottleneck,
            attn_head=attn_heads, attn_dim_head=attn_dim_head, act=act
        )
        self.mid_block2 = _ResnetBlock3D(
            mid_dim, mid_dim, groups=groups,
            use_full_attention=full_attn_bottleneck,
            attn_head=attn_heads, attn_dim_head=attn_dim_head, act=act
        )

        # Decoder
        self.ups = nn.ModuleList()
        for idx, ((dim_in, dim_out), attn_flag) in enumerate(
            zip(reversed(in_out), reversed(full_attn_paths))
        ):
            self.ups.append(nn.ModuleList([
                _ResnetBlock3D(dim_out + dim_in, dim_out, groups=groups,
                               use_full_attention=attn_flag,
                               attn_head=attn_heads,
                               attn_dim_head=attn_dim_head, act=act),
                _ResnetBlock3D(dim_out + dim_in, dim_out, groups=groups,
                               use_full_attention=attn_flag,
                               attn_head=attn_heads,
                               attn_dim_head=attn_dim_head, act=act),
                _Upsample3D(dim_out, dim_in),
            ]))

        # 输出头
        self.final_conv = nn.Sequential(
            _ResnetBlock3D(initial_dim * 2, initial_dim,
                           groups=groups, act=act),
            nn.Conv3d(initial_dim, out_channels, 1)
        )

    def forward(self, x):
        x = self.init_conv(x)
        r = x.clone()

        skips = []
        for resnet1, resnet2, downsample in self.downs:
            x = resnet1(x)
            skips.append(x)
            x = resnet2(x)
            skips.append(x)
            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_block2(x)

        for resnet1, resnet2, upsample in self.ups:
            x = resnet1(torch.cat([x, skips.pop()], dim=1))
            x = resnet2(torch.cat([x, skips.pop()], dim=1))
            x = upsample(x)

        return self.final_conv(torch.cat([x, r], dim=1))


# =============================================================
#  模型加载工具
# =============================================================

def load_fullCT_model(cfg: dict, device: torch.device):
    """加载 UNet3D（fullCT 路线）"""
    spatial_transformer = SpatialTransformer(
        size=cfg['fullCT_target_shape']
    ).to(device)

    ckpt_path = cfg['ckpt_fullCT']
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint不存在: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and 'model' in ckpt:
        state = ckpt['model']
    else:
        state = ckpt

    if any(k.startswith('enc1.') for k in state.keys()):
        model = UNet3D(in_channels=2, out_channels=3).to(device)
        model_kind = 'unet3d'
    else:
        model = TrainingUnet(
            problem_dimension='3D',
            input_channels=2,
            out_channels=3,
            initial_dim=4,
            dim_mults=(2, 4, 8, 16),
            groups=2,
            full_attn_paths=(None, None, None, None),
            full_attn_bottleneck=False,
            act='ReLU',
        ).to(device)
        model_kind = 'unet'

    model.load_state_dict(state)
    model.eval()
    print(f"✅ fullCT model ({model_kind}) loaded from {ckpt_path}")
    return model, spatial_transformer, model_kind


def load_marker_model(cfg: dict, device: torch.device):
    """加载 Unet（marker 路线）"""
    model = Unet(
        problem_dimension='3D',
        input_channels=cfg['unet_input_channels'],
        out_channels=cfg['unet_out_channels'],
        initial_dim=cfg['unet_initial_dim'],
        dim_mults=tuple(cfg['unet_dim_mults']),
        groups=cfg['unet_groups'],
        attn_dim_head=cfg.get('unet_attn_dim_head', 16),
        attn_heads=cfg['unet_attn_heads'],
        full_attn_paths=tuple(cfg['unet_full_attn_paths']),
        full_attn_bottleneck=cfg['unet_full_attn_bottleneck'],
    ).to(device)

    ckpt_path = cfg['ckpt_marker']
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint不存在: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and 'model' in ckpt:
        state = ckpt['model']
    else:
        state = ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"✅ Unet (marker) loaded from {ckpt_path}")
    return model


# =============================================================
#  推理：预测 Flow Field
# =============================================================

@torch.no_grad()
def predict_flow_fullCT(model, moving_t: torch.Tensor,
                         fixed_t: torch.Tensor) -> np.ndarray:
    """
    返回 flow_np: [3, D, H, W]  (numpy)
    """
    flow = model(moving_t, fixed_t)          # [1,3,D,H,W]
    return flow[0].cpu().numpy()


@torch.no_grad()
def predict_flow_marker(model, moving_t: torch.Tensor,
                         fixed_t: torch.Tensor) -> np.ndarray:
    """
    返回 flow_np: [3, D, H, W]  (numpy)
    """
    x    = torch.cat([moving_t, fixed_t], dim=1)   # [1,2,D,H,W]
    flow = model(x)                                  # [1,3,D,H,W]
    return flow[0].cpu().numpy()


# =============================================================
#  TRE 计算工具
# =============================================================

def resample_markers_to_model_space(markers: np.ndarray,
                                    original_shape,
                                    target_shape) -> np.ndarray:
    """
    markers: [N, 3]  (x, y, z) in original voxel space
    返回: [N, 3] in target voxel space
    """
    scale = np.array(target_shape) / np.array(original_shape)
    return markers * scale


def trilinear_sample_flow(flow: np.ndarray,
                           positions: np.ndarray) -> np.ndarray:
    """
    flow:      [3, D, H, W]
    positions: [N, 3]  → (x, y, z) 对应 (W, H, D) 轴
    返回:      [N, 3]  各位置的flow值
    """
    N = positions.shape[0]
    _, D, H, W = flow.shape
    sampled = np.zeros((N, 3), dtype=np.float32)

    for i, (px, py, pz) in enumerate(positions):
        # 注意：marker坐标顺序为 (x,y,z)=(W,H,D)
        x = np.clip(px, 0, W - 1.001)
        y = np.clip(py, 0, H - 1.001)
        z = np.clip(pz, 0, D - 1.001)

        x0, y0, z0 = int(x), int(y), int(z)
        x1 = min(x0 + 1, W - 1)
        y1 = min(y0 + 1, H - 1)
        z1 = min(z0 + 1, D - 1)

        xd, yd, zd = x - x0, y - y0, z - z0

        for c in range(3):
            c000 = flow[c, z0, y0, x0]
            c001 = flow[c, z0, y0, x1]
            c010 = flow[c, z0, y1, x0]
            c011 = flow[c, z0, y1, x1]
            c100 = flow[c, z1, y0, x0]
            c101 = flow[c, z1, y0, x1]
            c110 = flow[c, z1, y1, x0]
            c111 = flow[c, z1, y1, x1]

            c00 = c000 * (1 - xd) + c001 * xd
            c01 = c010 * (1 - xd) + c011 * xd
            c10 = c100 * (1 - xd) + c101 * xd
            c11 = c110 * (1 - xd) + c111 * xd

            c0 = c00 * (1 - yd) + c01 * yd
            c1 = c10 * (1 - yd) + c11 * yd

            sampled[i, c] = c0 * (1 - zd) + c1 * zd

    return sampled


def compute_tre(t00_markers: np.ndarray,
                t50_markers: np.ndarray,
                flow_np: np.ndarray,
                original_shape,
                target_shape) -> dict:
    """
    t00_markers, t50_markers: [N, 3] in original voxel space
    flow_np: [3, D, H, W] in model space

    返回包含所有TRE统计的字典
    """
    # 重采样到模型空间
    t00_rs = resample_markers_to_model_space(
        t00_markers, original_shape, target_shape)
    t50_rs = resample_markers_to_model_space(
        t50_markers, original_shape, target_shape)

    # Ground Truth 位移 = T00 - T50（把T50 warp到T00）
    gt_disp = t00_rs - t50_rs                         # [N, 3]

    # 从flow场采样预测位移（在T50位置采样）
    pred_disp = trilinear_sample_flow(flow_np, t50_rs) # [N, 3]

    # TRE = ||GT - Pred||_2
    error_vec = gt_disp - pred_disp                    # [N, 3]
    tre = np.linalg.norm(error_vec, axis=1)            # [N]

    gt_mag   = np.linalg.norm(gt_disp,   axis=1)
    pred_mag = np.linalg.norm(pred_disp, axis=1)

    corr = float(np.corrcoef(gt_mag, pred_mag)[0, 1]) \
        if len(gt_mag) > 1 else float('nan')

    return {
        'tre':           tre,
        'mean_tre':      float(tre.mean()),
        'std_tre':       float(tre.std()),
        'median_tre':    float(np.median(tre)),
        'max_tre':       float(tre.max()),
        'min_tre':       float(tre.min()),
        'gt_disp':       gt_disp,
        'pred_disp':     pred_disp,
        'gt_mag_mean':   float(gt_mag.mean()),
        'pred_mag_mean': float(pred_mag.mean()),
        'correlation':   corr,
    }


# =============================================================
#  图像质量指标（NCC / SSIM）
# =============================================================

def compute_image_metrics(moving_np: np.ndarray,
                           fixed_np:  np.ndarray,
                           warped_np: np.ndarray,
                           cfg: dict) -> dict:
    """
    moving_np, fixed_np, warped_np: [D, H, W]  numpy, 归一化到[0,1]
    """
    ncc_fn = NCC(win=cfg['ncc_win'])

    def to_t(arr):
        return torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)

    ncc_before = float(-ncc_fn(to_t(moving_np), to_t(fixed_np)).item())
    ncc_after  = float(-ncc_fn(to_t(warped_np), to_t(fixed_np)).item())

    # SSIM on one slice
    ax  = cfg['ssim_slice_axis']
    idx = cfg['ssim_slice_idx']

    def get_slice(vol):
        if ax == 0: return vol[idx, :, :]
        if ax == 1: return vol[:, idx, :]
        return vol[:, :, idx]

    def ssim_2d(a, b):
        data_range = max(b.max() - b.min(), 1e-6)
        return float(ssim_func(a, b, data_range=data_range))

    ssim_before = ssim_2d(get_slice(moving_np), get_slice(fixed_np))
    ssim_after  = ssim_2d(get_slice(warped_np), get_slice(fixed_np))

    return {
        'ncc_before':  ncc_before,
        'ncc_after':   ncc_after,
        'ncc_improve': ncc_after - ncc_before,
        'ssim_before': ssim_before,
        'ssim_after':  ssim_after,
        'ssim_improve':ssim_after - ssim_before,
    }


# =============================================================
#  可视化
# =============================================================

def visualize_case(case_name: str,
                   moving_np: np.ndarray,
                   fixed_np:  np.ndarray,
                   warped_np: np.ndarray,
                   flow_np:   np.ndarray,
                   tre_result: dict,
                   img_metrics: dict,
                   output_dir: str,
                   cfg: dict):
    """生成单个 Case 的汇总图"""

    os.makedirs(output_dir, exist_ok=True)
    ax  = cfg['ssim_slice_axis']
    idx = cfg['ssim_slice_idx']

    def get_slice(vol):
        if ax == 0: return vol[idx, :, :]
        if ax == 1: return vol[:, idx, :]
        return vol[:, :, idx]

    tre    = tre_result['tre']
    gt_d   = tre_result['gt_disp']
    pred_d = tre_result['pred_disp']

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'{case_name} — Registration Evaluation', fontsize=14, fontweight='bold')

    # Row 0: CT slices
    axes[0, 0].imshow(get_slice(moving_np), cmap='gray')
    axes[0, 0].set_title('Moving (T50)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(get_slice(fixed_np), cmap='gray')
    axes[0, 1].set_title('Fixed (T00)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(get_slice(warped_np), cmap='gray')
    axes[0, 2].set_title('Warped (Moving→Fixed)')
    axes[0, 2].axis('off')

    diff_before = np.abs(get_slice(moving_np) - get_slice(fixed_np))
    diff_after  = np.abs(get_slice(warped_np) - get_slice(fixed_np))
    vmax = max(diff_before.max(), diff_after.max())
    im = axes[0, 3].imshow(diff_after, cmap='hot', vmin=0, vmax=vmax)
    axes[0, 3].set_title('|Warped - Fixed|')
    axes[0, 3].axis('off')
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046)

    # Row 1: Flow magnitude + TRE
    flow_mag = np.sqrt((flow_np ** 2).sum(axis=0))
    im1 = axes[1, 0].imshow(
        flow_mag[idx] if ax == 0 else
        flow_mag[:, idx, :] if ax == 1 else
        flow_mag[:, :, idx],
        cmap='jet'
    )
    axes[1, 0].set_title('Flow Magnitude')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)

    # TRE bar per marker
    axes[1, 1].bar(range(len(tre)), tre,
                   color=['green' if t < 2.0 else 'red' for t in tre])
    axes[1, 1].axhline(y=1.0, color='blue', linestyle='--', alpha=0.7, label='1.0 vox')
    axes[1, 1].axhline(y=2.0, color='orange', linestyle='--', alpha=0.7, label='2.0 vox')
    axes[1, 1].set_title(f'TRE per Marker (mean={tre_result["mean_tre"]:.3f})')
    axes[1, 1].set_xlabel('Marker index')
    axes[1, 1].set_ylabel('TRE [voxels]')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(axis='y', alpha=0.3)

    # GT vs Pred scatter
    gt_mag_per   = np.linalg.norm(gt_d,   axis=1)
    pred_mag_per = np.linalg.norm(pred_d, axis=1)
    axes[1, 2].scatter(gt_mag_per, pred_mag_per,
                        c=tre, cmap='hot', s=80, edgecolors='k', linewidths=1)
    lim = max(gt_mag_per.max(), pred_mag_per.max()) * 1.1
    axes[1, 2].plot([0, lim], [0, lim], 'k--', alpha=0.5)
    axes[1, 2].set_xlabel('GT Magnitude [vox]')
    axes[1, 2].set_ylabel('Pred Magnitude [vox]')
    axes[1, 2].set_title(f'Magnitude (r={tre_result["correlation"]:.3f})')
    axes[1, 2].grid(alpha=0.3)

    # Component-wise error
    err_per_axis = np.abs(gt_d - pred_d)
    axes[1, 3].bar(['X', 'Y', 'Z'],
                   [err_per_axis[:, i].mean() for i in range(3)],
                   yerr=[err_per_axis[:, i].std()  for i in range(3)],
                   color=['#FF6B6B', '#4ECDC4', '#95E1D3'],
                   edgecolor='black', capsize=8)
    axes[1, 3].set_title('Mean Error per Axis')
    axes[1, 3].set_ylabel('Abs Error [voxels]')
    axes[1, 3].grid(axis='y', alpha=0.3)

    # Row 2: Metrics summary text
    for col in range(4):
        axes[2, col].axis('off')

    summary = (
        f"{'='*60}\n"
        f"  {case_name}  —  Summary\n"
        f"{'='*60}\n\n"
        f"  TRE (voxels)\n"
        f"    Mean   : {tre_result['mean_tre']:.3f} ± {tre_result['std_tre']:.3f}\n"
        f"    Median : {tre_result['median_tre']:.3f}\n"
        f"    Range  : {tre_result['min_tre']:.3f} – {tre_result['max_tre']:.3f}\n"
    )
    for th in cfg['tre_thresholds']:
        pct = (tre < th).mean() * 100
        summary += f"    < {th:.1f} vox : {pct:.1f}%\n"

    summary += (
        f"\n  Image Metrics\n"
        f"    NCC  before : {img_metrics['ncc_before']:.4f}\n"
        f"    NCC  after  : {img_metrics['ncc_after']:.4f}  "
        f"(Δ {img_metrics['ncc_improve']:+.4f})\n"
        f"    SSIM before : {img_metrics['ssim_before']:.4f}\n"
        f"    SSIM after  : {img_metrics['ssim_after']:.4f}  "
        f"(Δ {img_metrics['ssim_improve']:+.4f})\n"
        f"\n  Displacement\n"
        f"    GT   mean mag : {tre_result['gt_mag_mean']:.3f} vox\n"
        f"    Pred mean mag : {tre_result['pred_mag_mean']:.3f} vox\n"
        f"    Correlation   : {tre_result['correlation']:.4f}\n"
    )

    axes[2, 0].text(
        0.02, 0.95, summary,
        transform=axes[2, 0].transAxes,
        fontsize=9, family='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    )

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{case_name}_evaluation.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 图表已保存: {out_path}")


# =============================================================
#  保存结果
# =============================================================

def save_results(all_results: list, cfg: dict, output_dir: str, logger):
    """保存 JSON 摘要 + TXT 表格"""

    # --- JSON ---
    if cfg['save_json']:
        summary = []
        for r in all_results:
            entry = {k: v for k, v in r.items()
                     if not isinstance(v, np.ndarray)}
            summary.append(entry)
        json_path = os.path.join(output_dir, 'results_summary.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON saved: {json_path}")

    # --- TXT 表格 ---
    txt_path = os.path.join(output_dir, 'results_table.txt')
    header = (f"{'Case':<12} {'N_Mkr':<7} {'MeanTRE':<10} {'MedTRE':<10} "
              f"{'MaxTRE':<10} {'<2vox%':<9} {'NCC_imp':<10} {'SSIM_imp':<10} "
              f"{'Corr':<8}")
    sep = '-' * len(header)

    lines = [sep, header, sep]
    all_tre = []

    for r in all_results:
        tre = r['tre']
        all_tre.append(tre)
        pct2 = (tre < 2.0).mean() * 100
        lines.append(
            f"{r['case']:<12} {r['n_markers']:<7} "
            f"{r['mean_tre']:<10.3f} {r['median_tre']:<10.3f} "
            f"{r['max_tre']:<10.3f} {pct2:<9.1f} "
            f"{r['ncc_improve']:<10.4f} {r['ssim_improve']:<10.4f} "
            f"{r['correlation']:<8.4f}"
        )

    lines.append(sep)

    if all_tre:
        combined = np.concatenate(all_tre)
        lines.append(
            f"{'OVERALL':<12} {len(combined):<7} "
            f"{combined.mean():<10.3f} {np.median(combined):<10.3f} "
            f"{combined.max():<10.3f} "
            f"{(combined < 2.0).mean()*100:<9.1f} "
            f"{'—':<10} {'—':<10} {'—':<8}"
        )
        lines.append(sep)

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    logger.info(f"TXT table saved: {txt_path}")

    # 打印到控制台
    for line in lines:
        logger.info(line)


# =============================================================
#  单条 Pipeline：fullCT 路线
# =============================================================

def run_fullCT_pipeline(cfg: dict, logger) -> list:
    logger.info("=" * 60)
    logger.info("Pipeline: fullCT (UNet3D + Dataset_FullCT)")
    logger.info("=" * 60)

    device     = get_device(cfg)
    output_dir = os.path.join(cfg['output_root'], 'fullCT')
    os.makedirs(output_dir, exist_ok=True)

    model, spatial_transformer, model_kind = load_fullCT_model(cfg, device)
    case_dirs = get_case_list(cfg)
    logger.info(f"Test cases ({len(case_dirs)}): "
                f"{[os.path.basename(c) for c in case_dirs]}")

    dataset = Dataset_FullCT(
        case_dirs,
        nii_root        = cfg['nii_root'],
        target_shape    = tuple(cfg['fullCT_target_shape']),
        fixed_nii_name  = cfg['fixed_nii_name'],
        moving_nii_name = cfg['moving_nii_name'],
    )

    all_results = []

    for idx, case_dir in enumerate(case_dirs):
        case_name = os.path.basename(case_dir)
        logger.info(f"\n{'─'*50}")
        logger.info(f"Processing: {case_name}  [{idx+1}/{len(case_dirs)}]")

        # ── 推理 ──
        moving_t, fixed_t = dataset[idx]
        moving_t = moving_t.unsqueeze(0).to(device)
        fixed_t  = fixed_t.unsqueeze(0).to(device)

        if model_kind == 'unet3d':
            flow_np = predict_flow_fullCT(model, moving_t, fixed_t)
        else:
            flow_np = predict_flow_marker(model, moving_t, fixed_t)

        # ── Warp ──
        warped_t = spatial_transformer(moving_t, torch.from_numpy(
            flow_np[None]).to(device))
        moving_np = moving_t[0, 0].cpu().numpy()
        fixed_np  = fixed_t[0, 0].cpu().numpy()
        warped_np = warped_t[0, 0].cpu().numpy()

        # ── TRE ──
        t00_markers, t50_markers = load_markers(case_dir, cfg)
        fixed_rel = cfg['fixed_nii_name'].format(
            case_name=case_name,
            case_name_lower=case_name.lower()
        )
        fixed_candidates = [
            os.path.join(case_dir, fixed_rel),
            os.path.join(cfg['nii_root'], case_name, fixed_rel),
            os.path.join(cfg['nii_root'], case_name.lower(), fixed_rel),
        ]
        fixed_nii_path = next((p for p in fixed_candidates if os.path.exists(p)),
                              fixed_candidates[0])
        original_shape = nib.load(fixed_nii_path).shape

        tre_result = compute_tre(
            t00_markers, t50_markers,
            flow_np, original_shape,
            tuple(cfg['fullCT_target_shape'])
        )

        # ── 图像质量指标 ──
        img_metrics = compute_image_metrics(
            moving_np, fixed_np, warped_np, cfg)

        # ── 日志 ──
        logger.info(
            f"  TRE  mean={tre_result['mean_tre']:.3f} ± "
            f"{tre_result['std_tre']:.3f}  "
            f"median={tre_result['median_tre']:.3f}  "
            f"max={tre_result['max_tre']:.3f} voxels"
        )
        logger.info(
            f"  NCC  {img_metrics['ncc_before']:.4f} → "
            f"{img_metrics['ncc_after']:.4f}  "
            f"(Δ{img_metrics['ncc_improve']:+.4f})"
        )
        logger.info(
            f"  SSIM {img_metrics['ssim_before']:.4f} → "
            f"{img_metrics['ssim_after']:.4f}  "
            f"(Δ{img_metrics['ssim_improve']:+.4f})"
        )

        # ── 可视化 ──
        if cfg['save_figures']:
            visualize_case(
                case_name, moving_np, fixed_np, warped_np,
                flow_np, tre_result, img_metrics, output_dir, cfg
            )

        # ── 保存 npy ──
        if cfg['save_flow_npy']:
            np.save(os.path.join(output_dir, f'{case_name}_flow.npy'), flow_np)
        if cfg['save_warped_npy']:
            np.save(os.path.join(output_dir, f'{case_name}_warped.npy'), warped_np)

        # ── 汇总 ──
        result = {
            'case':       case_name,
            'n_markers':  len(tre_result['tre']),
            'tre':        tre_result['tre'],
            **{k: v for k, v in tre_result.items() if k != 'tre'},
            **img_metrics,
        }
        all_results.append(result)

    save_results(all_results, cfg, output_dir, logger)
    return all_results


# =============================================================
#  单条 Pipeline：marker 路线
# =============================================================

def run_marker_pipeline(cfg: dict, logger) -> list:
    logger.info("=" * 60)
    logger.info("Pipeline: marker (Unet + Dataset_MarkerCT)")
    logger.info("=" * 60)

    device     = get_device(cfg)
    output_dir = os.path.join(cfg['output_root'], 'marker')
    os.makedirs(output_dir, exist_ok=True)

    model = load_marker_model(cfg, device)
    case_dirs = get_case_list(cfg)
    logger.info(f"Test cases ({len(case_dirs)}): "
                f"{[os.path.basename(c) for c in case_dirs]}")

    dataset = Dataset_MarkerCT(
        case_dirs,
        image_size        = tuple(cfg['marker_image_size']),
        marker_ct_suffix  = cfg['marker_ct_suffix'],
    )

    # marker 路线没有独立的 SpatialTransformer，用 F.grid_sample 手动 warp
    target_shape = tuple(cfg['marker_image_size'])
    spatial_transformer = SpatialTransformer(size=target_shape).to(device)

    all_results = []

    for idx, case_dir in enumerate(case_dirs):
        case_name = os.path.basename(case_dir)
        logger.info(f"\n{'─'*50}")
        logger.info(f"Processing: {case_name}  [{idx+1}/{len(case_dirs)}]")

        # ── 推理 ──
        moving_t, fixed_t = dataset[idx]
        moving_t = moving_t.unsqueeze(0).to(device)
        fixed_t  = fixed_t.unsqueeze(0).to(device)

        flow_np = predict_flow_marker(model, moving_t, fixed_t)

        # ── Warp ──
        warped_t  = spatial_transformer(
            moving_t, torch.from_numpy(flow_np[None]).to(device))
        moving_np = moving_t[0, 0].cpu().numpy()
        fixed_np  = fixed_t[0, 0].cpu().numpy()
        warped_np = warped_t[0, 0].cpu().numpy()

        # ── TRE ──
        t00_markers, t50_markers = load_markers(case_dir, cfg)

        # marker 路线的 original_shape 来自 CT 文件本身
        t00_nii = os.path.join(
            case_dir,
            f"{case_name}_T00{cfg['marker_ct_suffix']}"
        )
        original_shape = nib.load(t00_nii).shape if os.path.exists(t00_nii) \
            else target_shape

        tre_result = compute_tre(
            t00_markers, t50_markers,
            flow_np, original_shape, target_shape
        )

        # ── 图像质量指标（归一化后计算）──
        def norm01(arr):
            mn, mx = arr.min(), arr.max()
            return (arr - mn) / (mx - mn + 1e-8)

        img_metrics = compute_image_metrics(
            norm01(moving_np), norm01(fixed_np), norm01(warped_np), cfg)

        # ── 日志 ──
        logger.info(
            f"  TRE  mean={tre_result['mean_tre']:.3f} ± "
            f"{tre_result['std_tre']:.3f}  "
            f"median={tre_result['median_tre']:.3f}  "
            f"max={tre_result['max_tre']:.3f} voxels"
        )
        logger.info(
            f"  NCC  {img_metrics['ncc_before']:.4f} → "
            f"{img_metrics['ncc_after']:.4f}  "
            f"(Δ{img_metrics['ncc_improve']:+.4f})"
        )

        # ── 可视化 ──
        if cfg['save_figures']:
            visualize_case(
                case_name, norm01(moving_np), norm01(fixed_np),
                norm01(warped_np), flow_np,
                tre_result, img_metrics, output_dir, cfg
            )

        # ── 保存 npy ──
        if cfg['save_flow_npy']:
            np.save(os.path.join(output_dir, f'{case_name}_flow.npy'), flow_np)
        if cfg['save_warped_npy']:
            np.save(os.path.join(output_dir, f'{case_name}_warped.npy'), warped_np)

        result = {
            'case':      case_name,
            'n_markers': len(tre_result['tre']),
            'tre':       tre_result['tre'],
            **{k: v for k, v in tre_result.items() if k != 'tre'},
            **img_metrics,
        }
        all_results.append(result)

    save_results(all_results, cfg, output_dir, logger)
    return all_results


# =============================================================
#  Main
# =============================================================

def main():
    output_dir = CONFIG['output_root']
    logger     = setup_logger(output_dir)

    logger.info("=" * 60)
    logger.info("CT Registration — Complete Evaluation Pipeline")
    logger.info(f"Mode       : {CONFIG['pipeline_mode']}")
    logger.info(f"Split      : {CONFIG['split_mode']}")
    logger.info(f"Output dir : {output_dir}")
    logger.info("=" * 60)

    mode = CONFIG['pipeline_mode']

    if mode in ('fullCT', 'both'):
        run_fullCT_pipeline(CONFIG, logger)

    if mode in ('marker', 'both'):
        run_marker_pipeline(CONFIG, logger)

    logger.info("\n✅  Pipeline completed!")
    logger.info(f"📁  Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
