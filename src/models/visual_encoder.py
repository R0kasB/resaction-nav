import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoV2Encoder(nn.Module):
    """
    Frozen DINOv2 visual encoder.

    Input:
        images: torch.Tensor with shape (C, H, W) or (B, C, H, W), values in [0, 1]

    Output:
        features: torch.Tensor with shape (B, output_dim)
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model_name = model_name
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval()
        self.model.to(self.device)

        for param in self.model.parameters():
            param.requires_grad = False

        self.output_dim = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536,
        }[model_name]

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 3:
            images = images.unsqueeze(0)

        if images.dim() != 4:
            raise ValueError(f"Expected image tensor with shape (C,H,W) or (B,C,H,W), got {images.shape}")

        images = images.to(self.device)

        images = F.interpolate(
            images,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        images = (images - self.mean) / self.std

        features = self.model(images)

        if isinstance(features, dict):
            features = features["x_norm_clstoken"]

        return features