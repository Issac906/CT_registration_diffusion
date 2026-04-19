import numpy as np
import torch

import CT_registration_diffusion.Generator as Generator
import CT_registration_diffusion.model.predict_engine as predict_engine


class TwoStage4DCTRegistration(object):
    """
    Minimal coarse-to-local 4DCT registration scaffold.
    Stage 1 predicts a whole-volume coarse DVF.
    Stage 2 refines the coarse warped image with a local patch model.
    """

    def __init__(
        self,
        global_model,
        local_model,
        patch_size,
        patch_stride=None,
        device='cuda',
    ):
        self.global_model = global_model
        self.local_model = local_model
        self.patch_size = list(patch_size)
        self.patch_stride = list(patch_stride if patch_stride is not None else patch_size)
        self.device = device

    def build_global_dataset(self, **dataset_kwargs):
        return Generator.Dataset_4DCT(**dataset_kwargs)

    def build_local_dataset(self, **dataset_kwargs):
        return Generator.PatchCascadeDataset(patch_size=self.patch_size, **dataset_kwargs)

    def build_global_predictor(self, generator, batch_size=1):
        return predict_engine.Predictor(
            self.global_model,
            generator,
            batch_size=batch_size,
            device=self.device,
        )

    def build_local_predictor(self):
        return predict_engine.PatchPredictor(
            self.local_model,
            patch_size=self.patch_size,
            stride=self.patch_stride,
            device=self.device,
        )

    def run_global_stage(self, generator, trained_model_filename, batch_size=1):
        global_predictor = self.build_global_predictor(generator, batch_size=batch_size)
        pred_dvf, pred_dvf_numpy, coarse_warped_numpy = global_predictor.predict_MVF_and_apply(
            trained_model_filename=trained_model_filename
        )
        return {
            'coarse_dvf_tensor': pred_dvf,
            'coarse_dvf_numpy': pred_dvf_numpy,
            'coarse_warped_image': coarse_warped_numpy,
        }

    def run_local_stage(self, coarse_warped_image, fixed_image, trained_model_filename):
        local_predictor = self.build_local_predictor()
        residual_dvf_volume, refined_volume, patch_metadata = local_predictor.refine_volume(
            coarse_warped_image=coarse_warped_image,
            fixed_image=fixed_image,
            trained_model_filename=trained_model_filename,
        )
        return {
            'local_residual_dvf': residual_dvf_volume,
            'refined_warped_image': refined_volume,
            'patch_metadata': patch_metadata,
        }

    def run_two_stage_inference(
        self,
        moving_image,
        fixed_image,
        global_model_path,
        local_model_path,
    ):
        moving_tensor = self._to_tensor(moving_image)
        fixed_tensor = self._to_tensor(fixed_image)

        global_predictor = predict_engine.Predictor(
            self.global_model,
            generator=_SinglePairGenerator(moving_image, fixed_image),
            batch_size=1,
            device=self.device,
        )
        coarse_dvf, coarse_dvf_numpy, coarse_warped_numpy = global_predictor.predict_MVF_and_apply(
            trained_model_filename=global_model_path
        )

        local_output = self.run_local_stage(
            coarse_warped_image=coarse_warped_numpy,
            fixed_image=fixed_image,
            trained_model_filename=local_model_path,
        )

        return {
            'coarse_dvf_tensor': coarse_dvf,
            'coarse_dvf_numpy': coarse_dvf_numpy,
            'coarse_warped_image': coarse_warped_numpy,
            'local_residual_dvf': local_output['local_residual_dvf'],
            'refined_warped_image': local_output['refined_warped_image'],
            'patch_metadata': local_output['patch_metadata'],
            'moving_tensor_shape': tuple(moving_tensor.shape),
            'fixed_tensor_shape': tuple(fixed_tensor.shape),
        }

    def _to_tensor(self, image):
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 4:
            tensor = tensor.unsqueeze(0)
        return tensor


class _SinglePairGenerator(torch.utils.data.Dataset):
    """Tiny adapter so the existing Predictor can run on in-memory volumes."""

    def __init__(self, moving_image, fixed_image):
        self.moving_image = np.expand_dims(np.asarray(moving_image, dtype=np.float32), axis=0)
        self.fixed_image = np.expand_dims(np.asarray(fixed_image, dtype=np.float32), axis=0)
        self.image_size = list(self.moving_image.shape[1:])
        self.background_cutoff = 0.0
        self.maximum_cutoff = 1.0
        self.normalize_factor = 1.0

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.moving_image, self.fixed_image
