import math

import torch
import torch.nn.functional as F


def degrade_resolution(img: torch.Tensor, level: int) -> torch.Tensor:
    """
    img: (C, H, W) or (1, C, H, W), values in [0, 1]; H and W must be divisible by 2^level.
    level: integer k >= 0, block size = 2^k; max valid level = floor(log2(min(H, W))).
    returns: tensor with same spatial shape, each 2^k × 2^k block replaced by its average.
    """
    squeezed = False
    if img.dim() == 3:
        img = img.unsqueeze(0)
        squeezed = True

    _, _, H, W = img.shape
    max_level = int(math.floor(math.log2(min(H, W))))

    if level < 0 or level > max_level:
        raise ValueError(f"level must be between 0 and {max_level} for {H}×{W} images, got {level}")

    if level == 0:
        out = img
    else:
        block_size = 2 ** level
        pooled = F.avg_pool2d(img, kernel_size=block_size, stride=block_size)
        out = F.interpolate(pooled, size=(H, W), mode="nearest")

    if squeezed:
        out = out.squeeze(0)

    return out
