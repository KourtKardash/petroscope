import argparse
from pathlib import Path

import petroscope.segmentation.models as models
from petroscope.segmentation.classes import LumenStoneClasses
from petroscope.segmentation import SegmDetailedTester
from petroscope.utils.base import prepare_experiment


def get_test_img_mask_pairs(ds_dir: Path):
    """
    Get paths to test images and corresponding masks from dataset directory.
    """
    test_img_mask_p = [
        (img_p, ds_dir / "masks" / "test" / f"{img_p.stem}.png")
        for img_p in sorted((ds_dir / "imgs" / "test").iterdir())
    ]
    return test_img_mask_p


def run_test(
    classes_name: str,
    ds_dir: Path,
    out_dir: Path,
    device: str,
    void_pad=4,
    void_border_width=2,
    vis_segmentation=True,
):
    """
    Runs model on test images from dataset directory and
    saves results to output directory.
    """
    classes = LumenStoneClasses.from_name(classes_name)
    # create the model (ResUNet, PSPNet, HRNet) and load weights
    # model = models.HRNet.from_pretrained(
    #     "s1s2_w18_x05",
    #     device,
    #     force_download=True,
    # )
    model = models.ResUNet.from_pretrained(
        # "s1s2_resnet34_x05",
        Path.home()
        / "dev/petroscope/petroscope/segmentation/models/outputs/seed_experiments/resunet"
        / "ps384_combined_40/models/best_val_loss_weights.pth",
        # / "ps384_combined_40/models/best_test_miou_weights.pth",
        # / "dev/petroscope/petroscope/segmentation/models/outputs/ipta_experiments/pspnet_best"
        # / "models/best_val_loss_weights.pth",
        device,
        force_download=True,
    )

    tester = SegmDetailedTester(
        out_dir=out_dir,
        classes=classes,
        void_pad=void_pad,
        void_border_width=void_border_width,
        vis_segmentation=vis_segmentation,
    )
    res, res_void = tester.test_on_set(
        get_test_img_mask_pairs(ds_dir),
        predict_func=model.predict_image,
        epoch=0,
    )
    print("results without void borders:\n", res)
    print("results with void borders:\n", res_void)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Device to use for inference",
    )
    args = parser.parse_args()
    run_test(
        classes_name="S1_S2",
        ds_dir=Path.home() / "dev/LumenStone/S1v2_S2v2_x05",
        out_dir=prepare_experiment(Path("./out")),
        device=args.device,
    )
