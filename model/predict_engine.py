import sys 
sys.path.append('/host/d/Github/')

import math
import copy
import os
import pandas as pd
import numpy as np
import nibabel as nb
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
from skimage.measure import block_reduce

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

import CT_registration_diffusion.functions_collection as ff
import CT_registration_diffusion.Data_processing as Data_processing
import CT_registration_diffusion.patch_sampling as patch_sampling
import CT_registration_diffusion.model.model as my_model
import CT_registration_diffusion.model.spatial_transform as spatial_transform



class Predictor(object):
    def __init__(
        self,
        model,
        generator,
        batch_size,
        device = 'cuda',

    ):
        super().__init__()

        # model
        self.model = model  
        if device == 'cuda':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == 'cpu':
            self.device = torch.device("cpu")

        self.image_size = generator.image_size
        self.batch_size = batch_size

        # dataset and dataloader
        self.generator = generator
        dl = DataLoader(self.generator, batch_size = self.batch_size, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())    
        self.background_cutoff = self.generator.background_cutoff
        self.maximum_cutoff = self.generator.maximum_cutoff
        self.normalize_factor = self.generator.normalize_factor

        self.dl = dl
 
        # EMA:
        self.ema = EMA(model)
        self.ema.to(self.device)

    def load_model(self, trained_model_filename):

        data = torch.load(trained_model_filename, map_location=self.device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.ema.load_state_dict(data["ema"])

    
    def predict_MVF_and_apply(self, trained_model_filename):
        self.load_model(trained_model_filename)
        self.ema.ema_model.eval()
        with torch.inference_mode():
            for batch_input in tqdm(self.dl):
                moving_image, fixed_image = batch_input
                data_moving  = moving_image.to(self.device)
                data_fixed = fixed_image.to(self.device)
                data_input = torch.cat((data_moving, data_fixed), dim=1)

                pred_MVF = self.ema.ema_model(data_input)

                pred_MVF_numpy = torch.clone(pred_MVF).detach().cpu().numpy().squeeze()

                # apply the MVF to moving image to get the warped image
                warped_moving_image = spatial_transform.warp_from_mvf(data_moving, pred_MVF)
                warped_moving_image_numpy = warped_moving_image.detach().cpu().numpy().squeeze()

                # de-normalize the warped image
                warped_moving_image_numpy = Data_processing.normalize_image(warped_moving_image_numpy, normalize_factor =self.generator.normalize_factor, image_max = self.generator.maximum_cutoff, image_min = self.generator.background_cutoff, invert = True, final_max = 1, final_min = 0)

                
            return pred_MVF, pred_MVF_numpy, warped_moving_image_numpy


class PatchPredictor(object):
    """
    Minimal local patch predictor.
    It refines a coarse warped volume patch-by-patch and blends the refined
    patches back into a full-size volume.
    """

    def __init__(
        self,
        model,
        patch_size,
        stride=None,
        device='cuda',
        blend_sigma_scale=0.125,
    ):
        self.model = model
        self.patch_size = list(patch_size)
        self.stride = list(stride if stride is not None else patch_size)
        self.blend_sigma_scale = float(blend_sigma_scale)
        if device == 'cuda':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.ema = EMA(model)
        self.ema.to(self.device)
        self.patch_sampler = patch_sampling.VolumePatchSampler(
            patch_size=self.patch_size,
            stride=self.stride,
        )
        self.blend_weight_patch = self._build_blend_weight(self.patch_size)

    def load_model(self, trained_model_filename):
        data = torch.load(trained_model_filename, map_location=self.device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.ema.load_state_dict(data["ema"])

    def _numpy_to_tensor(self, image):
        tensor = torch.from_numpy(image.astype(np.float32))
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 4:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    def _build_blend_weight(self, patch_size):
        axes = []
        for size in patch_size:
            if size <= 1:
                axes.append(np.ones((1,), dtype=np.float32))
                continue

            center = (size - 1) / 2.0
            sigma = max(size * self.blend_sigma_scale, 1.0)
            coord = np.arange(size, dtype=np.float32)
            axis_weight = np.exp(-0.5 * ((coord - center) / sigma) ** 2)
            axis_weight = axis_weight / np.max(axis_weight)
            axes.append(axis_weight.astype(np.float32))

        weight = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
        weight = np.maximum(weight, 1e-6)
        return weight.astype(np.float32)

    def refine_volume(self, coarse_warped_image, fixed_image, trained_model_filename=None):
        if trained_model_filename is not None:
            self.load_model(trained_model_filename)

        coarse_warped_image = np.asarray(coarse_warped_image, dtype=np.float32)
        fixed_image = np.asarray(fixed_image, dtype=np.float32)
        assert coarse_warped_image.shape == fixed_image.shape, "coarse and fixed image shapes must match"

        refined_sum = np.zeros_like(coarse_warped_image, dtype=np.float32)
        residual_dvf_sum = np.zeros((3,) + coarse_warped_image.shape, dtype=np.float32)
        weight_sum = np.zeros_like(coarse_warped_image, dtype=np.float32)
        patch_metadata = []

        self.ema.ema_model.eval()
        with torch.inference_mode():
            for patch_record in self.patch_sampler.iter_sliding(
                coarse_warped_image,
                fixed_image,
                coarse_pad_value=np.min(coarse_warped_image),
                target_pad_value=np.min(fixed_image),
            ):
                coarse_patch = patch_record.coarse_patch
                fixed_patch = patch_record.target_patch

                coarse_patch_tensor = self._numpy_to_tensor(coarse_patch)
                fixed_patch_tensor = self._numpy_to_tensor(fixed_patch)
                patch_input = torch.cat((coarse_patch_tensor, fixed_patch_tensor), dim=1)

                residual_dvf_patch = self.ema.ema_model(patch_input)
                refined_patch = spatial_transform.warp_from_mvf(coarse_patch_tensor, residual_dvf_patch)

                residual_dvf_patch_np = residual_dvf_patch.detach().cpu().numpy().squeeze(0)
                refined_patch_np = refined_patch.detach().cpu().numpy().squeeze()

                slices = patch_record.valid_slices
                patch_shape = tuple(s.stop - s.start for s in slices)
                patch_crop = tuple(slice(0, size) for size in patch_shape)
                blend_weight = self.blend_weight_patch[patch_crop]
                patch_metadata.append(patch_record.to_dict())

                refined_sum[slices] += refined_patch_np[patch_crop] * blend_weight
                residual_dvf_sum[(slice(None),) + slices] += residual_dvf_patch_np[
                    (slice(None),) + patch_crop
                ] * blend_weight[None, ...]
                weight_sum[slices] += blend_weight

        weight_sum = np.maximum(weight_sum, 1e-6)
        refined_volume = refined_sum / weight_sum
        residual_dvf_volume = residual_dvf_sum / weight_sum[None, ...]
        return residual_dvf_volume, refined_volume, patch_metadata
