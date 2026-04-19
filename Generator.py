# generator

import sys 
sys.path.append('/host/d/Github/')
import os
import torch
import numpy as np
import nibabel as nb
import random
import re
import glob
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

import CT_registration_diffusion.functions_collection as ff
import CT_registration_diffusion.Data_processing as Data_processing
import CT_registration_diffusion.patch_sampling as patch_sampling

# define augmentation functions here if needed
# random functionf
def random_rotate(i, z_rotate_degree = None, z_rotate_range = [-5,5], fill_val = None, order = 1):
    # only do rotate according to z (in-plane rotation)
    if z_rotate_degree is None:
        z_rotate_degree = random.uniform(z_rotate_range[0], z_rotate_range[1])

    if fill_val is None:
        fill_val = np.min(i)
    
    if z_rotate_degree == 0:
        return i, z_rotate_degree
    else:
        if len(i.shape) == 2:
            return Data_processing.rotate_image(np.copy(i), z_rotate_degree, order = order, fill_val = fill_val, ), z_rotate_degree
        else:
            return Data_processing.rotate_image(np.copy(i), [0,0,z_rotate_degree], order = order, fill_val = fill_val, ), z_rotate_degree

def random_translate(i, x_translate = None,  y_translate = None, translate_range = [-5,5]):
    # only do translate according to x and y
    if x_translate is None or y_translate is None:
        x_translate = int(random.uniform(translate_range[0], translate_range[1]))
        y_translate = int(random.uniform(translate_range[0], translate_range[1]))
    
    if len(i.shape) == 2:
        return Data_processing.translate_image(np.copy(i), [x_translate,y_translate]), x_translate,y_translate
    else:
        return Data_processing.translate_image(np.copy(i), [x_translate,y_translate,0]), x_translate,y_translate


# class Dataset_4DCT(Dataset):
#     def __init__(
#         self,
#         image_folder_list,
#         image_size = [224,224,96], # target image size after center-crop

#         num_of_pairs_each_case = 1, # number of image pairs to be sampled from each 4DCT case
#         preset_paired_tf = None, # preset paired time frames if needed, e.g., [[0,3],[1,2]], otherwise randomly pick two time frames
#         only_use_tf0_as_moving = None, # if set True, only use time frame 0 as moving image, otherwise randomly select moving time frame

#         cutoff_range = [-200,250], # default cutoff range for CT images
#         normalize_factor = 'equatoin',
#         shuffle = False,

#         augment = True, # whether to do data augmentation
#         augment_frequency = 0.5, # frequency of augmentation
      
#     ):
#         super().__init__()
#         self.image_folder_list = image_folder_list
#         self.image_size = image_size

#         self.num_of_pairs_each_case = num_of_pairs_each_case
#         self.preset_paired_tf = preset_paired_tf
#         if self.preset_paired_tf is not None:
#             assert self.num_of_pairs_each_case == len(self.preset_paired_tf)
#         self.only_use_tf0_as_moving = only_use_tf0_as_moving; assert self.only_use_tf0_as_moving in [True, False]
       
#         self.background_cutoff = cutoff_range[0]
#         self.maximum_cutoff = cutoff_range[1]
#         self.normalize_factor = normalize_factor

#         self.shuffle = shuffle
#         self.augment = augment
#         self.augment_frequency = augment_frequency

#         self.num_files = len(image_folder_list)

#         self.index_array = self.generate_index_array()
#         # self.current_moving_file = None
#         # self.current_moving_data = None
#         # self.current_fixed_file = None
#         # self.current_fixed_data = None
       

#     def generate_index_array(self): 
#         np.random.seed()
#         index_array = []
        
#         # loop through all files
#         if self.shuffle == True:
#             f_list = np.random.permutation(self.num_files)
#         else:
#             f_list = np.arange(self.num_files)
        
#         for f in f_list:
#             # loop through all pairs in each file 
#             for p in range(self.num_of_pairs_each_case):
#                 index_array.append([f,p])
      
#         return index_array

#     def __len__(self):
#        return self.num_files * self.num_of_pairs_each_case
    
#     def load_data(self, file_path):
#         image = nb.load(file_path).get_fdata()
#         return image

  
#     def __getitem__(self, index):
#         file_index, pair_index = self.index_array[index]
#         current_image_folder = self.image_folder_list[file_index]
        
#         # randomly pick two time frames or using preset paired time frames
#         timeframes = ff.find_all_target_files(['img*'], current_image_folder)
#         if self.preset_paired_tf is not None:
#             t1, t2 = self.preset_paired_tf[pair_index]
#             # print('这里的time frame配对是预设的,不是随机选取的, pick time frames:', t1, t2)
#             if self.only_use_tf0_as_moving == True:
#                 assert t1 == 0, 'when only_use_tf0_as_moving is set True, the preset paired time frames must have time frame 0 as moving image'
#         else:
#             if self.only_use_tf0_as_moving == True:
#                 t1 = 0
#                 t2 = np.random.choice([i for i in range(len(timeframes)) if i != t1])
#             else:
#                 t1, t2 = np.random.choice(len(timeframes), size=2, replace=False)
#             # print('这里的time frame配对是随机选取的, pick time frames:', t1, t2)
#         moving_file = timeframes[t1]
#         fixed_file = timeframes[t2]
#         # print('in this folder, I pick moving file:', moving_file, ' fixed file:', fixed_file)

#         # load image
#         moving_image = self.load_data(moving_file)
#         fixed_image = self.load_data(fixed_file)

#         # augmentation for noise if needed
#         if self.augment == True and (np.random.rand() < self.augment_frequency):
#             # add noise, make sure the noise added to both images are the same
#             standard_deviation = np.random.uniform(5,15) # standard deviation of the noise
#             noise = np.random.normal(0, standard_deviation, moving_image.shape)
#             moving_image = moving_image + noise
#             fixed_image = fixed_image + noise

#         # preprocess if needed
#         # cutoff 
#         # print('before cutoff, image range:', np.min(moving_image), np.max(moving_image))
#         if self.background_cutoff is not None and self.maximum_cutoff is not None:
#             moving_image = Data_processing.cutoff_intensity(moving_image, self.background_cutoff, self.maximum_cutoff)
#             fixed_image = Data_processing.cutoff_intensity(fixed_image, self.background_cutoff, self.maximum_cutoff)
      
#         # normalization to [-1,1]
#         moving_image = Data_processing.normalize_image(moving_image, normalize_factor = self.normalize_factor, image_max = self.maximum_cutoff, image_min = self.background_cutoff ,invert = False,final_max = 1, final_min = 0)
#         fixed_image = Data_processing.normalize_image(fixed_image, normalize_factor = self.normalize_factor, image_max = self.maximum_cutoff, image_min = self.background_cutoff ,invert = False,final_max = 1, final_min = 0)
#         # print('after cutoff and normalization, image range:', np.min(moving_image), np.max(moving_image))
        
#         # augmentation if needed
#         # step 2: rotate [-10,10] degrees according to z-axis
#         # step 3: translate [-10,10] pixels
#         if self.augment == True and (np.random.rand() < self.augment_frequency):

#             # rotate (according to z-axis), make sure rotate angle is the same for both images, use function random_rotate above
#             moving_image, z_rotate_degree = random_rotate(moving_image,  order = 1)
#             moving_image, x_translate, y_translate = random_translate(moving_image)
#             fixed_image, _ = random_rotate(fixed_image, z_rotate_degree, order = 1)
#             fixed_image, _, _ = random_translate(fixed_image, x_translate, y_translate)

#         # print('after preprocessing and augmentation, image shape:', moving_image.shape)

#         # make it a standard dimension for deep learning model: [channel, x,y,z]
#         moving_image = np.expand_dims(moving_image, axis=0)  # add channel dimension
#         fixed_image = np.expand_dims(fixed_image, axis=0)  # add channel

#         return moving_image, fixed_image
# def on_epoch_end(self):
#         self.index_array = self.generate_index_array()
    
    

class Dataset_4DCT(Dataset):
    def __init__(
        self,
        image_folder_list,
        image_size=[224, 224, 96],

        num_of_pairs_each_case=1,
        preset_paired_tf=None,
        only_use_tf0_as_moving=None,

        cutoff_range=[-200, 250],
        normalize_factor='equatoin',
        shuffle=False,

        augment=True,
        augment_frequency=0.5,

        # ✅ 新增：支持 cascade
        stage=1,                 # 1 / 2 / 3
        warped_root=None,        # stage 2/3 时必须传
    ):
        super().__init__()
        self.image_folder_list = image_folder_list
        self.image_size = image_size

        self.num_of_pairs_each_case = num_of_pairs_each_case
        self.preset_paired_tf = preset_paired_tf
        if self.preset_paired_tf is not None:
            assert self.num_of_pairs_each_case == len(self.preset_paired_tf)

        # ✅ 修：允许 None（否则你不传就炸）
        self.only_use_tf0_as_moving = only_use_tf0_as_moving
        if self.only_use_tf0_as_moving is None:
            self.only_use_tf0_as_moving = False
        assert self.only_use_tf0_as_moving in [True, False]

        self.background_cutoff = cutoff_range[0]
        self.maximum_cutoff = cutoff_range[1]
        self.normalize_factor = normalize_factor

        self.shuffle = shuffle
        self.augment = augment
        self.augment_frequency = augment_frequency

        self.num_files = len(image_folder_list)

        # ✅ 新增：cascade 参数保存
        self.stage = stage
        self.warped_root = warped_root
        if self.stage in [2, 3]:
            assert self.warped_root is not None, "stage 2/3 requires warped_root (path to stage1/2 warped outputs)"

        self.index_array = self.generate_index_array()

    def generate_index_array(self):
        np.random.seed()
        index_array = []

        if self.shuffle is True:
            f_list = np.random.permutation(self.num_files)
        else:
            f_list = np.arange(self.num_files)

        for f in f_list:
            for p in range(self.num_of_pairs_each_case):
                index_array.append([f, p])

        return index_array

    def __len__(self):
        return self.num_files * self.num_of_pairs_each_case

    def load_data(self, file_path):
        image = nb.load(file_path).get_fdata()
        return image

    def _parse_tf_index(self, filepath: str) -> int:
        """Extract time-frame index from filenames like img_0.nii or img_0.nii.gz"""
        name = os.path.basename(filepath)
        m = re.search(r"img_(\d+)", name)
        if m is None:
            raise ValueError(f"Cannot parse time-frame index from filename: {name}")
        return int(m.group(1))

    def _warped_path(self, current_image_folder: str, fixed_file: str, tf_index: int) -> str:
        """Locate warped file for stage 2/3."""
        # current_image_folder is often ".../CaseX/cropped_image"
        folder = current_image_folder.rstrip("/\\")
        base = os.path.basename(folder)                      # e.g. "cropped_image"
        parent = os.path.basename(os.path.dirname(folder))   # e.g. "Case1"

        # pick Case folder name, not "cropped_image"
        case_id = parent if base.lower() in ["cropped_image", "cropped", "images", "image", "img"] else base

        # 1) warped_root/<case_id>/warped_tfX.nii.gz
        cand1 = os.path.join(self.warped_root, case_id, f"warped_tf{tf_index}.nii.gz")
        if os.path.exists(cand1):
            return cand1

        # 2) warped_root/<case_id>/epoch_*/warped_tfX.nii.gz
        epoch_hits = glob.glob(os.path.join(self.warped_root, case_id, 'epoch_*', f"warped_tf{tf_index}.nii.gz"))
        if len(epoch_hits) > 0:
            epoch_hits = sorted(epoch_hits)
            return epoch_hits[-1]

        # 3) warped_root/warped_tfX.nii.gz （如果 warped_root 已经指到 case 或 epoch 里）
        cand2 = os.path.join(self.warped_root, f"warped_tf{tf_index}.nii.gz")
        if os.path.exists(cand2):
            return cand2

        # 4) fallback: warped_root/<case_id>/<basename(fixed_file)>
        cand3 = os.path.join(self.warped_root, case_id, os.path.basename(fixed_file))
        if os.path.exists(cand3):
            return cand3

        raise FileNotFoundError(f"Cannot find warped file for case={case_id}, tf={tf_index}. Tried: {cand1}, {cand2}, {cand3}")

    def __getitem__(self, index):
        file_index, pair_index = self.index_array[index]
        current_image_folder = self.image_folder_list[file_index]

        timeframes = ff.find_all_target_files(['img*'], current_image_folder)

        # pick paired timeframes
        if self.preset_paired_tf is not None:
            t1, t2 = self.preset_paired_tf[pair_index]
            if self.only_use_tf0_as_moving is True:
                assert t1 == 0, "only_use_tf0_as_moving=True requires preset pairs to have t1=0"
            if self.stage in [2, 3]:
                assert t2 != 0
        else:
            if self.only_use_tf0_as_moving is True:
                t1 = 0
                if self.stage in [2, 3]:
                    t2 = np.random.choice([i for i in range(len(timeframes)) if i != 0])
                else:
                    t2 = np.random.choice([i for i in range(len(timeframes)) if i != t1])
            else:
                if self.stage in [2, 3]:
                    t2 = np.random.choice([i for i in range(len(timeframes)) if i != 0])
                    t1 = np.random.choice([i for i in range(len(timeframes)) if i != t2])
                else:
                    t1, t2 = np.random.choice(len(timeframes), size=2, replace=False)

        moving_file = timeframes[t1]
        fixed_file = timeframes[t2]

        # ✅ cascade：stage2/3 强制 tf1=tf2，moving 从 warped_root 读
        if self.stage in [2, 3]:
            tf_index = self._parse_tf_index(fixed_file)              # ✅ 修：函数名
            moving_path = self._warped_path(current_image_folder, fixed_file, tf_index)  # ✅ 修：函数名
            fixed_path = fixed_file
        else:
            moving_path = moving_file
            fixed_path = fixed_file

        # ✅ 修：真正读数据时必须用 moving_path / fixed_path
        moving_image = self.load_data(moving_path)
        fixed_image = self.load_data(fixed_path)

        # noise augmentation
        if self.augment is True and (np.random.rand() < self.augment_frequency):
            standard_deviation = np.random.uniform(5, 15)
            noise = np.random.normal(0, standard_deviation, moving_image.shape)
            moving_image = moving_image + noise
            fixed_image = fixed_image + noise

        # cutoff
        if self.background_cutoff is not None and self.maximum_cutoff is not None:
            moving_image = Data_processing.cutoff_intensity(moving_image, self.background_cutoff, self.maximum_cutoff)
            fixed_image = Data_processing.cutoff_intensity(fixed_image, self.background_cutoff, self.maximum_cutoff)

        # normalize
        moving_image = Data_processing.normalize_image(
            moving_image, normalize_factor=self.normalize_factor,
            image_max=self.maximum_cutoff, image_min=self.background_cutoff,
            invert=False, final_max=1, final_min=0
        )
        fixed_image = Data_processing.normalize_image(
            fixed_image, normalize_factor=self.normalize_factor,
            image_max=self.maximum_cutoff, image_min=self.background_cutoff,
            invert=False, final_max=1, final_min=0
        )

        # geometric augmentation
        if self.augment is True and (np.random.rand() < self.augment_frequency):
            moving_image, z_rotate_degree = random_rotate(moving_image, order=1)
            moving_image, x_translate, y_translate = random_translate(moving_image)
            fixed_image, _ = random_rotate(fixed_image, z_rotate_degree, order=1)
            fixed_image, _, _ = random_translate(fixed_image, x_translate, y_translate)

        moving_image = np.expand_dims(moving_image, axis=0)
        fixed_image = np.expand_dims(fixed_image, axis=0)

        print("STAGE", self.stage, "moving_path:", moving_path, "fixed_path:", fixed_path)

        return moving_image, fixed_image

    def on_epoch_end(self):
        self.index_array = self.generate_index_array()


class PatchCascadeDataset(Dataset):
    """
    Minimal local-refinement dataset.
    It reads a coarse warped image and the matching fixed image, then samples
    corresponding 3D patches from both volumes.
    """

    def __init__(
        self,
        image_folder_list,
        warped_root,
        patch_size,
        patches_per_pair=1,
        image_size=[224, 224, 96],
        num_of_pairs_each_case=1,
        preset_paired_tf=None,
        only_use_tf0_as_moving=True,
        cutoff_range=[-200, 250],
        normalize_factor='equatoin',
        shuffle=False,
        augment=False,
        augment_frequency=0.0,
        fixed_patch_starts=None,
    ):
        super().__init__()
        self.image_folder_list = image_folder_list
        self.warped_root = warped_root
        self.patch_size = list(patch_size)
        self.patches_per_pair = int(patches_per_pair)
        if self.patches_per_pair < 1:
            raise ValueError("patches_per_pair must be >= 1")
        self.image_size = image_size
        self.num_of_pairs_each_case = num_of_pairs_each_case
        self.preset_paired_tf = preset_paired_tf
        self.only_use_tf0_as_moving = only_use_tf0_as_moving
        self.background_cutoff = cutoff_range[0]
        self.maximum_cutoff = cutoff_range[1]
        self.normalize_factor = normalize_factor
        self.shuffle = shuffle
        self.augment = augment
        self.augment_frequency = augment_frequency
        self.fixed_patch_starts = fixed_patch_starts or {}
        self.patch_sampler = patch_sampling.VolumePatchSampler(patch_size=self.patch_size)

        self.num_files = len(image_folder_list)
        self.index_array = self.generate_index_array()

    def generate_index_array(self):
        np.random.seed()
        file_indices = np.random.permutation(self.num_files) if self.shuffle else np.arange(self.num_files)
        index_array = []
        for file_index in file_indices:
            for pair_index in range(self.num_of_pairs_each_case):
                for patch_index in range(self.patches_per_pair):
                    index_array.append([file_index, pair_index, patch_index])
        return index_array

    def __len__(self):
        return len(self.index_array)

    def load_data(self, file_path):
        return nb.load(file_path).get_fdata()

    def _parse_tf_index(self, filepath):
        name = os.path.basename(filepath)
        match = re.search(r"img_(\d+)", name)
        if match is None:
            raise ValueError(f"Cannot parse timeframe from {name}")
        return int(match.group(1))

    def _case_id_from_folder(self, image_folder):
        folder = image_folder.rstrip("/\\")
        base = os.path.basename(folder)
        if base.lower() in ["cropped_image", "cropped", "images", "image", "img"]:
            case_id = os.path.basename(os.path.dirname(folder))
            dataset_id = os.path.basename(os.path.dirname(os.path.dirname(folder)))
        else:
            case_id = base
            dataset_id = os.path.basename(os.path.dirname(folder))
        return f"{dataset_id}_{case_id}"

    def _resolve_warped_path(self, image_folder, tf_index):
        case_id = self._case_id_from_folder(image_folder)

        candidates = [
            os.path.join(self.warped_root, case_id, f"warped_tf{tf_index}.nii.gz"),
            os.path.join(self.warped_root, f"warped_tf{tf_index}.nii.gz"),
        ]

        epoch_hits = glob.glob(os.path.join(self.warped_root, case_id, "epoch_*", f"warped_tf{tf_index}.nii.gz"))
        if epoch_hits:
            candidates.append(sorted(epoch_hits)[-1])

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(f"Cannot find coarse warped image for case={case_id}, tf={tf_index}")

    def _select_pair(self, timeframes, pair_index):
        if self.preset_paired_tf is not None:
            return self.preset_paired_tf[pair_index]

        if self.only_use_tf0_as_moving:
            moving_tf = 0
            fixed_tf = np.random.choice([i for i in range(len(timeframes)) if i != moving_tf])
            return moving_tf, fixed_tf

        moving_tf, fixed_tf = np.random.choice(len(timeframes), size=2, replace=False)
        return moving_tf, fixed_tf

    def _preprocess_image(self, image):
        if self.background_cutoff is not None and self.maximum_cutoff is not None:
            image = Data_processing.cutoff_intensity(image, self.background_cutoff, self.maximum_cutoff)
        image = Data_processing.normalize_image(
            image,
            normalize_factor=self.normalize_factor,
            image_max=self.maximum_cutoff,
            image_min=self.background_cutoff,
            invert=False,
            final_max=1,
            final_min=0,
        )
        return image

    def _sample_patch_start(self, case_id, tf_index, volume_shape):
        key = (case_id, tf_index)
        if key in self.fixed_patch_starts:
            return tuple(self.fixed_patch_starts[key])
        return self.patch_sampler.random_start(volume_shape)

    def __getitem__(self, index):
        file_index, pair_index, patch_index = self.index_array[index]
        image_folder = self.image_folder_list[file_index]
        case_id = self._case_id_from_folder(image_folder)

        timeframes = ff.find_all_target_files(['img*'], image_folder)
        moving_tf, fixed_tf = self._select_pair(timeframes, pair_index)
        fixed_path = timeframes[fixed_tf]
        coarse_path = self._resolve_warped_path(image_folder, fixed_tf)

        coarse_image = self._preprocess_image(self.load_data(coarse_path))
        fixed_image = self._preprocess_image(self.load_data(fixed_path))

        patch_start = self._sample_patch_start(case_id, fixed_tf, fixed_image.shape)
        patch_record = self.patch_sampler.sample_random(
            coarse_image,
            fixed_image,
            start=patch_start,
            coarse_pad_value=np.min(coarse_image),
            target_pad_value=np.min(fixed_image),
        )
        coarse_patch = patch_record.coarse_patch
        fixed_patch = patch_record.target_patch

        if self.augment and (np.random.rand() < self.augment_frequency):
            noise_std = np.random.uniform(5, 15)
            noise = np.random.normal(0, noise_std, coarse_patch.shape)
            coarse_patch = coarse_patch + noise
            fixed_patch = fixed_patch + noise

        coarse_patch = np.expand_dims(coarse_patch, axis=0).astype(np.float32)
        fixed_patch = np.expand_dims(fixed_patch, axis=0).astype(np.float32)
        metadata = {
            "case_id": case_id,
            "fixed_tf": fixed_tf,
            "moving_tf": moving_tf,
            "patch_index": patch_index,
            "coarse_path": coarse_path,
            "fixed_path": fixed_path,
            "patch_start": patch_record.start,
            "patch_end": patch_record.end,
            "patch_center": patch_record.center,
            "valid_slices": patch_record.valid_slices,
            "volume_shape": patch_record.volume_shape,
            "patch_size": patch_record.patch_size,
        }
        return coarse_patch, fixed_patch, metadata

    def on_epoch_end(self):
        self.index_array = self.generate_index_array()
