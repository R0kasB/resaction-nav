from pathlib import Path
import numpy as np
import torch
import torchvision

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from ..utils.image_resolution import degrade_resolution
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
    Path("output").mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cam = ThorCamera(scene="FloorPlan1", width=256, height=256, device=device)

    try:

        obs0 = cam.get_observation(level=0)
        obs2 = cam.get_observation(level=2)

        torchvision.utils.save_image(obs0, "output/obs_level0.png")
        torchvision.utils.save_image(obs2, "output/obs_level2.png")


    finally:
        cam.close()


if __name__ == "__main__":
    main()