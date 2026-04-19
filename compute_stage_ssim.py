import sys
import argparse
import os
from pathlib import Path

import nibabel as nb
import numpy as np
from skimage.metrics import structural_similarity

sys.path.append("/host/d/GitHub")

import CT_registration_diffusion.Data_processing as Data_processing


def parse_args():
    parser = argparse.ArgumentParser(description="Compute SSIM for stage1/2/3 warped outputs against GT.")
    parser.add_argument("--save-root", default="/host/d/projects/registration/models/trial_1")
    parser.add_argument("--dataset", default="DIR_LAB", choices=["DIR_LAB", "Popi"])
    parser.add_argument("--case", default="Case1")
    parser.add_argument("--tf", type=int, default=3, help="Target timeframe index, e.g. 3 -> img_3 / warped_tf3")
    parser.add_argument("--image-max", type=float, default=250)
    parser.add_argument("--image-min", type=float, default=-200)
    parser.add_argument("--normalize-factor", default="equation")
    parser.add_argument("--win-size", type=int, default=7, help="SSIM window size. Must be odd.")
    return parser.parse_args()


def latest_epoch_dir(stage_root: Path, case_id: str) -> Path:
    case_root = stage_root / case_id
    if not case_root.exists():
        raise FileNotFoundError(f"Missing case folder: {case_root}")

    epoch_dirs = [p for p in case_root.iterdir() if p.is_dir() and p.name.startswith("epoch_")]
    if not epoch_dirs:
        raise FileNotFoundError(f"No epoch folders found under: {case_root}")

    return max(epoch_dirs, key=lambda p: int(p.name.split("_")[-1]))


def load_and_normalize_nii(path: Path, normalize_factor, image_max, image_min):
    img = nb.load(str(path)).get_fdata()
    img = Data_processing.normalize_image(
        img,
        normalize_factor=normalize_factor,
        image_max=image_max,
        image_min=image_min,
        invert=False,
        final_max=1,
        final_min=0,
    )
    return img.astype(np.float32)


def compute_ssim_3d(img1: np.ndarray, img2: np.ndarray, win_size: int) -> float:
    return float(
        structural_similarity(
            img1,
            img2,
            data_range=1.0,
            win_size=win_size,
        )
    )


def main():
    args = parse_args()

    if args.win_size % 2 == 0:
        raise ValueError("--win-size must be odd")

    save_root = Path(args.save_root)
    gt_path = Path(f"/host/d/Data/CT_image/{args.dataset}/{args.case}/cropped_image/img_{args.tf}.nii.gz")
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing GT image: {gt_path}")

    gt_img = load_and_normalize_nii(
        gt_path,
        normalize_factor=args.normalize_factor,
        image_max=args.image_max,
        image_min=args.image_min,
    )

    stage_map = {
        "stage1": save_root / "results_stage1",
        "stage2": save_root / "results_stage2",
        "stage3": save_root / "results_stage3",
    }

    print(f"GT: {gt_path}")
    print(f"Case: {args.case}, tf: {args.tf}, dataset: {args.dataset}")

    for stage_name, stage_root in stage_map.items():
        epoch_dir = latest_epoch_dir(stage_root, args.case)
        warped_path = epoch_dir / f"warped_tf{args.tf}.nii.gz"
        if not warped_path.exists():
            raise FileNotFoundError(f"Missing warped image: {warped_path}")

        warped_img = load_and_normalize_nii(
            warped_path,
            normalize_factor=args.normalize_factor,
            image_max=args.image_max,
            image_min=args.image_min,
        )

        ssim_value = compute_ssim_3d(warped_img, gt_img, win_size=args.win_size)
        print(f"{stage_name} | epoch={epoch_dir.name} | SSIM={ssim_value:.6f} | warped={warped_path}")


if __name__ == "__main__":
    main()
