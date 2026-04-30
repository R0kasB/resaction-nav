import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from pathlib import Path

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering


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


# ---------- AI2-THOR camera wrapper ----------

class ThorCamera:
    """
    Handles:
      - AI2-THOR Controller
      - grabbing RGB frames from the environment
      - applying resolution levels via degrade_resolution()

    Typical usage:
      cam = ThorCamera(scene="FloorPlan1")
      obs = cam.get_observation(level=2)  # torch.Tensor (C, 256, 256), float in [0, 1]
    """

    def __init__(
        self,
        scene: str = "FloorPlan1",
        width: int = 256,
        height: int = 256,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)

        self.controller = Controller(
            scene=scene,
            platform=CloudRendering, 
            width=width,
            height=height,
        )

        event = self.controller.last_event
        frame = event.frame  # NumPy array (H, W, 3), uint8
        h, w, _ = frame.shape
        if (w, h) != (width, height):
            raise RuntimeError(
                f"Unexpected frame size {w}x{h}, expected {width}x{height}"
            )

    def close(self) -> None:
        self.controller.stop()

    def _frame_to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """
        Convert event.frame (H, W, 3) uint8 -> torch (C, 256, 256) float in [0, 1]
        """
        
        frame = np.array(frame, copy=True)

        # (H, W, 3) -> (3, H, W)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return tensor.to(self.device)

    def get_raw_frame(self) -> torch.Tensor:
        """
        Return raw camera image without resolution degradation:
        shape (C, 256, 256), float in [0, 1]
        """
        event = self.controller.last_event
        frame = event.frame
        img = self._frame_to_tensor(frame)
        return img

    def get_observation(self, level: int) -> torch.Tensor:
        """
        Return the observation for the RL agent:
          - grab frame from env
          - convert to tensor (C, 256, 256) float in [0, 1]
          - apply resolution degradation for the given level
        """
        img = self.get_raw_frame()
        obs = degrade_resolution(img, level=level)
        return obs

    def step(self, action: str, level: int) -> torch.Tensor:
        """
        Example method:
          - apply an AI2-THOR action
          - return observation at the given resolution level

        action: e.g. "MoveAhead", "RotateRight", "RotateLeft", ...
        """
        event = self.controller.step(action=action)
        frame = event.frame
        img = self._frame_to_tensor(frame)
        obs = degrade_resolution(img, level=level)
        return obs



def main():
    cam = ThorCamera(scene="FloorPlan1", width=256, height=256, device="cuda")

    try:

        obs0 = cam.get_observation(level=0)
        obs2 = cam.get_observation(level=2)

        torchvision.utils.save_image(obs0, "output/obs_level0.png")
        torchvision.utils.save_image(obs2, "output/obs_level2.png")


    finally:
        cam.close()


if __name__ == "__main__":
    main()
    