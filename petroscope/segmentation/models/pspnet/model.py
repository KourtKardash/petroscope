from pathlib import Path
from typing import Any

from petroscope.segmentation.models.base import PatchSegmentationModel
from petroscope.utils import logger
from petroscope.utils.lazy_imports import torch  # noqa
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)


class PSPNet(PatchSegmentationModel):

    MODEL_REGISTRY: dict[str, str] = {
        "s1s2_resnet34_x05": (
            "http://www.xubiker.online/petroscope/segmentation_weights"
            "/pspnet_resnet34/S1v2_S2v2_x05.pth"
        ),
    }

    def __init__(
        self, n_classes: int, backbone: str, dilated: bool, device: str, n_rotated: int | None = None
    ) -> None:
        """
        Initialize the PSPNet model.

        Args:
            n_classes: Number of segmentation classes
            backbone: Backbone network (e.g., resnet18)
            dilated: Whether to use dilated convolutions
            device: Device to run the model on ('cuda', 'cpu', etc.)
        """
        super().__init__(n_classes, device, n_rotated)

        from petroscope.segmentation.models.pspnet.nn import PSPNet

        self.backbone = backbone
        self.dilated = dilated
        self.n_rotated = n_rotated
        # Determine appropriate weights based on backbone
        weights = None
        if backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT
        elif backbone == "resnet34":
            weights = ResNet34_Weights.DEFAULT
        elif backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT
        elif backbone == "resnet101":
            weights = ResNet101_Weights.DEFAULT
        elif backbone == "resnet152":
            weights = ResNet152_Weights.DEFAULT

        self.model = PSPNet(
            n_classes=n_classes,
            dilated=dilated,
            backbone=backbone,
            weights=weights,
            n_rotated = self.n_rotated
        ).to(self.device)

    def _get_checkpoint_data(self) -> dict[str, Any]:
        """Return model-specific data for checkpoint saving."""
        return {
            "n_classes": self.n_classes,
            "backbone": self.backbone,
            "dilated": self.dilated,
            "n_add_imgs": self.n_rotated
        }

    @classmethod
    def _create_from_checkpoint(
        cls, checkpoint: dict, device: str
    ) -> "PSPNet":
        """Create a PSPNet model from checkpoint data."""
        return cls(
            n_classes=checkpoint["n_classes"],
            backbone=checkpoint["backbone"],
            dilated=checkpoint["dilated"],
            device=device,
            add_imgs=checkpoint.get("add_imgs", None)
        )
