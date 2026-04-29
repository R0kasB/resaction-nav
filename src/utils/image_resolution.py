import torch
import torch.nn.functional as F

def degrade_resolution(img: torch.Tensor, level: int) -> torch.Tensor:
    """
    img: torch.Tensor of shape (C, 256, 256) or (1, C, 256, 256), values in [0, 1]
    level: integer k >= 0, block size = 2^k (for 256x256, k <= 8)
    returns: tensor with same spatial shape, where each 2^k x 2^k block is constant
    """
    if level < 0 or level > 8:
        raise ValueError("level must be between 0 and 8 for 256x256 images")

    block_size = 2 ** level

    squeezed = False
    if img.dim() == 3:
        # (C, H, W) -> (1, C, H, W)
        img = img.unsqueeze(0)
        squeezed = True

    if img.shape[-2:] != (256, 256):
        raise ValueError(f"Expected image size (256, 256), got {img.shape[-2:]}")

    # Average 2^k x 2^k blocks (downsample)
    pooled = F.avg_pool2d(img, kernel_size=block_size, stride=block_size)

    # Replicate each block back to 256x256 (upsample)
    out = F.interpolate(pooled, size=(256, 256), mode="nearest")

    if squeezed:
        out = out.squeeze(0)

    return out