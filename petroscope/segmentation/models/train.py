from pathlib import Path
from typing import Literal
import warnings

import argparse
import copy
from dataclasses import dataclass
import hydra
from omegaconf import DictConfig
import sys

from petroscope.segmentation.balancer import ClassBalancedPatchDataset
from petroscope.segmentation.classes import LumenStoneClasses
from petroscope.segmentation.models.base import PatchSegmentationModel
from petroscope.segmentation.models.resunet.model import ResUNet
from petroscope.segmentation.models.pspnet.model import PSPNet
from petroscope.segmentation.models.hrnet import HRNet
from petroscope.segmentation.utils import BasicBatchCollector
from petroscope.utils import logger

seed = 1
model_type = 'resunet'

@dataclass
class AnisotropicParams:
    anisotropic_mode: bool = False
    add_img_dir_path: str = None
    n_rotated: int = None
    step_polazied: int = None

anisotropic_params = AnisotropicParams()

def set_pytorch_seed(seed: int):
    """Fix all pytorch seeds for reproducibility."""
    import os

    # Set environment variable for PyTorch to enable deterministic algorithms
    # (should be set before importing torch)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    from petroscope.utils.lazy_imports import torch  # noqa

    # Set PyTorch's random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.__version__ >= "1.8.0":
        # Use deterministic algorithms where available (PyTorch 1.8+)
        # This is a warning-only setting and will not raise an error
        # if deterministic algorithms are not available
        # Suppress warning for operations without deterministic implementation
        warnings.filterwarnings(
            "ignore",
            message=".*nll_loss2d_forward_out_cuda_template does not have a deterministic implementation.*",
        )
        torch.use_deterministic_algorithms(True, warn_only=True)


def test_img_mask_pairs(cfg: DictConfig):
    """Get test image-mask pairs from the dataset."""
    ds_dir = Path(cfg.data.dataset_path)
    test_img_mask_p = [
        (img_p, ds_dir / "masks" / "test" / f"{img_p.stem}.png")
        for img_p in sorted((ds_dir / "imgs" / "test").iterdir())
    ]
    return test_img_mask_p


def create_train_dataset(cfg: DictConfig, anisotropic_params: AnisotropicParams):
    """Create a training dataset.

    Args:
        cfg: Configuration object

    Returns:
        ClassBalancedPatchDataset instance
    """
    ds_dir = Path(cfg.data.dataset_path)
    train_img_mask_p = [
        (img_p, ds_dir / "masks" / "train" / f"{img_p.stem}.png")
        for img_p in sorted((ds_dir / "imgs" / "train").iterdir())
    ]

    return ClassBalancedPatchDataset(
        img_mask_paths=train_img_mask_p,
        patch_size=cfg.train.patch_size,
        augment_rotation=cfg.train.augm.rotation,
        augment_scale=cfg.train.augm.scale,
        augment_brightness=cfg.train.augm.brightness,
        augment_color=cfg.train.augm.color,
        class_set=LumenStoneClasses.from_name(cfg.data.classes),
        class_area_consideration=cfg.train.balancer.class_area_consideration,
        patch_positioning_accuracy=cfg.train.balancer.patch_positioning_accuracy,
        balancing_strength=cfg.train.balancer.balancing_strength,
        acceleration=cfg.train.balancer.acceleration,
        cache_dir=None,
        void_border_width=cfg.train.balancer.void_border_width,
        seed=cfg.hardware.seed,
        add_img_dir_path=anisotropic_params.add_img_dir_path,
        n_rotated=anisotropic_params.n_rotated,
        step_polazied=anisotropic_params.step_polazied,
        mode=anisotropic_params.anisotropic_mode
    )


def create_samplers(
    dataset: ClassBalancedPatchDataset,
    cfg: DictConfig,
    use_dataloaders: bool = True,
):
    """Create training and validation data samplers from a dataset.

    Args:
        dataset: The dataset to create samplers from
        cfg: Configuration object
        use_dataloaders: Whether to use PyTorch DataLoaders

    Returns:
        Tuple of (train_iterator, val_iterator, dataset_length)
    """
    balanced = cfg.train.balancer.enabled
    logger.info(f"Using {'balanced' if balanced else 'random'} sampling")

    train_it = None
    val_it = None

    if use_dataloaders:
        device = cfg.hardware.device
        use_pin_memory = cfg.train.data_loader.pin_memory

        # Setup pin_memory_device when using CUDA
        pin_memory_device = None
        if use_pin_memory and "cuda:" in device:
            # Extract device index for better compatibility
            device_idx = int(device.split(":")[-1])
            pin_memory_device = f"cuda:{device_idx}"
            logger.info(
                f"Using pin_memory with device: {device} (index: {device_idx})"
            )

        # Create dataloaders with proper device config
        train_sampler = (
            dataset.dataloader_balanced(
                batch_size=cfg.train.batch_size,
                num_workers=cfg.train.data_loader.num_workers,
                prefetch_factor=cfg.train.data_loader.prefetch_factor,
                pin_memory=use_pin_memory,
                pin_memory_device=pin_memory_device,
            )
            if balanced
            else dataset.dataloader_random(
                batch_size=cfg.train.batch_size,
                num_workers=cfg.train.data_loader.num_workers,
                prefetch_factor=cfg.train.data_loader.prefetch_factor,
                pin_memory=use_pin_memory,
                pin_memory_device=pin_memory_device,
            )
        )

        val_sampler = dataset.dataloader_random(
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.data_loader.num_workers,
            prefetch_factor=cfg.train.data_loader.prefetch_factor,
            pin_memory=use_pin_memory,
            pin_memory_device=pin_memory_device,
        )

        train_it = iter(train_sampler)
        val_it = iter(val_sampler)
    else:
        train_sampler = (
            dataset.sampler_balanced()
            if balanced
            else dataset.sampler_random()
        )
        val_sampler = dataset.sampler_random()

        train_it = iter(
            BasicBatchCollector(
                train_sampler,
                cfg.train.batch_size,
            )
        )
        val_it = iter(
            BasicBatchCollector(
                val_sampler,
                cfg.train.batch_size,
            )
        )

    return (train_it, val_it, len(dataset))


def create_model(
    model_type: Literal["resunet", "pspnet", "hrnet"],
    cfg: DictConfig,
    n_classes: int,
    n_rotated: int | None
) -> PatchSegmentationModel:
    """
    Create a segmentation model based on the specified type.

    Args:
        model_type: Type of model to create ("resunet", "pspnet" or "hrnet")
        cfg: Configuration object
        n_classes: Number of segmentation classes

    Returns:
        Initialized model
    """
    if model_type == "resunet":
        return ResUNet(
            n_classes=n_classes,
            device=cfg.hardware.device,
            filters=cfg.model.resunet.filters,
            layers=cfg.model.resunet.layers,
            backbone=cfg.model.resunet.get("backbone", None),
            dilated=cfg.model.resunet.get("dilated", False),
            pretrained=cfg.model.resunet.get("pretrained", True),
            n_rotated=n_rotated
        )
    elif model_type == "pspnet":
        return PSPNet(
            n_classes=n_classes,
            backbone=cfg.model.pspnet.backbone,
            dilated=cfg.model.pspnet.dilated,
            device=cfg.hardware.device,
            n_rotated=n_rotated
        )
    elif model_type == "hrnet":
        return HRNet(
            n_classes=n_classes,
            device=cfg.hardware.device,
            backbone=cfg.model.hrnet.backbone,
            pretrained=cfg.model.hrnet.pretrained,
            ocr_mid_channels=cfg.model.hrnet.ocr_mid_channels,
            dropout=cfg.model.hrnet.dropout,
            use_aux_head=cfg.model.hrnet.use_aux_head,
            n_rotated=n_rotated
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_loss_manager(cfg: DictConfig, dataset, device: str):
    """
    Create a loss manager based on the configuration.

    Args:
        cfg: Configuration object
        dataset: Dataset with get_class_pixel_counts method
        device: Device for tensor operations

    Returns:
        Initialized LossManager

    Raises:
        ValueError: If loss configuration is missing or invalid
    """
    from petroscope.segmentation.losses import LossManager

    # Get loss configuration
    loss_config = cfg.train.get("loss")
    if loss_config is None:
        raise ValueError("Loss configuration is required in training config")

    # Get class pixel counts from dataset
    class_counts = dataset.get_class_pixel_counts()
    logger.info(f"Using class counts for loss weighting: {class_counts}")

    return LossManager.from_config_and_dataset(loss_config, dataset, device)


@hydra.main(version_base="1.2", config_path=".", config_name="config.yaml")
def run_training(cfg: DictConfig):
    """
    Main training function.

    Args:
        cfg: Hydra configuration
    """

    # Set the random seed for reproducibility
    set_pytorch_seed(seed)

    # Get model type from config
    #model_type = cfg["model_type"]

    # Load class definitions
    classes = LumenStoneClasses.from_name(cfg.data.classes)

    # Create the training dataset
    train_ds = create_train_dataset(cfg, anisotropic_params)

    # Create data samplers using the dataset
    train_iterator, val_iterator, ds_len = create_samplers(train_ds, cfg)

    # Create model
    model = create_model(model_type, cfg, len(classes), anisotropic_params.n_rotated)

    logger.info(f"Training {model_type.upper()} model")
    logger.info(model.n_params_str)

    # Get loss configuration
    loss_config = cfg.train.get("loss", None)

    # Get class counts for loss weight calculation
    class_counts = (
        train_ds.get_class_pixel_counts()
        if hasattr(train_ds, "get_class_pixel_counts")
        else None
    )

    # Train the model
    model.train(
        train_iterator=train_iterator,
        val_iterator=val_iterator,
        n_steps=ds_len // cfg.train.batch_size * cfg.train.augm.factor,
        LR=cfg.train.LR,
        scheduler_patience=cfg.train.scheduler_patience,
        epochs=cfg.train.epochs,
        val_steps=cfg.train.val_steps,
        test_every=cfg.train.test_every,
        test_params=model.TestParams(
            classes=classes,
            img_mask_paths=test_img_mask_pairs(cfg),
            void_pad=cfg.test.void_pad,
            void_border_width=cfg.test.void_border_width,
            vis_segmentation=cfg.test.vis_segmentation,
            max_epoch_visualizations=cfg.test.max_epoch_visualizations,
        ),
        out_dir=Path("Results/Exp1"),
        amp=cfg.get("amp", False),
        gradient_clipping=cfg.get("gradient_clipping", 1.0),
        loss_config=loss_config,
        class_counts=class_counts,  # Pass class counts instead of the dataset
    )


def parse_args():
    original_argv = sys.argv.copy()
    
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument('--anisotropic', action='store_true')
    temp_args, _ = temp_parser.parse_known_args()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--model_type', type=str)
    
    if temp_args.anisotropic:
        parser.add_argument('--anisotropic', action='store_true')
        parser.add_argument('add_img_dir_path', type=str)
        parser.add_argument('n_rotated', type=int)
        parser.add_argument('step_polazied', type=int)
    
    args = parser.parse_args()

    if hasattr(args, 'add_img_dir_path'):
        params = AnisotropicParams(
            enabled=True,
            add_img_dir_path=args.add_img_dir_path,
            n_rotated=args.n_rotated,
            step_polazied=args.step_polazied
        )
    else:
        params = AnisotropicParams()
    
    sys.argv = [sys.argv[0]]
    
    return args.seed, args.model_type, params

if __name__ == "__main__":
    seed, model_type, params = parse_args()
    anisotropic_params = copy.copy(params)

    print(f"Seed: {seed}")
    print(f"Model type: {model_type}")
    print(f"Anisotropic params: {anisotropic_params}")
    print(f"Cleaned sys.argv: {sys.argv}")

    run_training()