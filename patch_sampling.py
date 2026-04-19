from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _as_shape(value, ndim: int) -> Tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * ndim
    value = tuple(int(v) for v in value)
    if len(value) != ndim:
        raise ValueError(f"Expected shape with {ndim} dims, got {value}")
    return value


@dataclass
class PatchRecord:
    coarse_patch: np.ndarray
    target_patch: np.ndarray
    start: Tuple[int, int, int]
    end: Tuple[int, int, int]
    center: Tuple[int, int, int]
    valid_slices: Tuple[slice, slice, slice]
    volume_shape: Tuple[int, int, int]
    patch_size: Tuple[int, int, int]

    def to_dict(self) -> Dict[str, object]:
        return {
            "coarse_patch": self.coarse_patch,
            "target_patch": self.target_patch,
            "patch_start": self.start,
            "patch_end": self.end,
            "patch_center": self.center,
            "valid_slices": self.valid_slices,
            "volume_shape": self.volume_shape,
            "patch_size": self.patch_size,
        }


class VolumePatchSampler:
    """
    Reusable 3D patch sampler for coarse-to-local registration.
    Training: random patch sampling from aligned coarse/target volumes.
    Inference: deterministic sliding-window traversal with recorded coordinates.
    """

    def __init__(self, patch_size: Sequence[int], stride: Optional[Sequence[int]] = None):
        self.patch_size = _as_shape(patch_size, 3)
        self.stride = _as_shape(stride if stride is not None else patch_size, 3)

    def sample_random(
        self,
        coarse_volume: np.ndarray,
        target_volume: np.ndarray,
        start: Optional[Sequence[int]] = None,
        coarse_pad_value: Optional[float] = None,
        target_pad_value: Optional[float] = None,
    ) -> PatchRecord:
        self._validate_inputs(coarse_volume, target_volume)
        if start is None:
            start = self._random_start(coarse_volume.shape)
        return self._build_patch_record(
            coarse_volume,
            target_volume,
            start=tuple(int(v) for v in start),
            coarse_pad_value=coarse_pad_value,
            target_pad_value=target_pad_value,
        )

    def random_start(self, volume_shape: Sequence[int]) -> Tuple[int, int, int]:
        return self._random_start(volume_shape)

    def iter_sliding(
        self,
        coarse_volume: np.ndarray,
        target_volume: np.ndarray,
        coarse_pad_value: Optional[float] = None,
        target_pad_value: Optional[float] = None,
    ) -> Iterable[PatchRecord]:
        self._validate_inputs(coarse_volume, target_volume)
        for start in self._iter_starts(coarse_volume.shape):
            yield self._build_patch_record(
                coarse_volume,
                target_volume,
                start=start,
                coarse_pad_value=coarse_pad_value,
                target_pad_value=target_pad_value,
            )

    def _validate_inputs(self, coarse_volume: np.ndarray, target_volume: np.ndarray) -> None:
        if coarse_volume.ndim != 3 or target_volume.ndim != 3:
            raise ValueError("VolumePatchSampler expects 3D volumes")
        if coarse_volume.shape != target_volume.shape:
            raise ValueError(f"Volume shapes must match, got {coarse_volume.shape} and {target_volume.shape}")

    def _random_start(self, volume_shape: Sequence[int]) -> Tuple[int, int, int]:
        starts: List[int] = []
        for dim, size in zip(volume_shape, self.patch_size):
            if dim <= size:
                starts.append(0)
            else:
                starts.append(int(np.random.randint(0, dim - size + 1)))
        return tuple(starts)

    def _iter_starts(self, volume_shape: Sequence[int]) -> Iterable[Tuple[int, int, int]]:
        ranges: List[List[int]] = []
        for dim, size, step in zip(volume_shape, self.patch_size, self.stride):
            if dim <= size:
                ranges.append([0])
                continue

            axis_starts = list(range(0, dim - size + 1, step))
            if axis_starts[-1] != dim - size:
                axis_starts.append(dim - size)
            ranges.append(axis_starts)

        for x in ranges[0]:
            for y in ranges[1]:
                for z in ranges[2]:
                    yield (x, y, z)

    def _build_patch_record(
        self,
        coarse_volume: np.ndarray,
        target_volume: np.ndarray,
        start: Tuple[int, int, int],
        coarse_pad_value: Optional[float],
        target_pad_value: Optional[float],
    ) -> PatchRecord:
        coarse_pad_value = float(np.min(coarse_volume)) if coarse_pad_value is None else coarse_pad_value
        target_pad_value = float(np.min(target_volume)) if target_pad_value is None else target_pad_value

        coarse_patch, valid_slices = self._extract_patch(coarse_volume, start, coarse_pad_value)
        target_patch, _ = self._extract_patch(target_volume, start, target_pad_value)

        end = tuple(start_axis + size for start_axis, size in zip(start, self.patch_size))
        center = tuple(start_axis + size // 2 for start_axis, size in zip(start, self.patch_size))
        return PatchRecord(
            coarse_patch=coarse_patch.astype(np.float32),
            target_patch=target_patch.astype(np.float32),
            start=start,
            end=end,
            center=center,
            valid_slices=valid_slices,
            volume_shape=tuple(int(v) for v in coarse_volume.shape),
            patch_size=self.patch_size,
        )

    def _extract_patch(
        self,
        volume: np.ndarray,
        start: Tuple[int, int, int],
        pad_value: float,
    ) -> Tuple[np.ndarray, Tuple[slice, slice, slice]]:
        patch = np.full(self.patch_size, pad_value, dtype=volume.dtype)

        src_slices = []
        dst_slices = []
        for axis_start, axis_size, dim in zip(start, self.patch_size, volume.shape):
            axis_end = axis_start + axis_size
            src_start = max(axis_start, 0)
            src_end = min(axis_end, dim)
            dst_start = max(0, -axis_start)
            dst_end = dst_start + max(0, src_end - src_start)
            src_slices.append(slice(src_start, src_end))
            dst_slices.append(slice(dst_start, dst_end))

        patch[tuple(dst_slices)] = volume[tuple(src_slices)]
        return patch, tuple(src_slices)
