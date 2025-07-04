from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from petroscope.segmentation.classes import LumenStoneClasses


def color_to_mask(
    mask_colored: np.ndarray, classes: LumenStoneClasses
) -> np.ndarray:
    mask = np.zeros(mask_colored.shape[:2], dtype=np.uint8)
    diffs = [
        np.sum(np.abs(mask_colored - cls.color_rgb), axis=-1)
        for cls in classes
    ]
    mask_idx = np.stack(diffs, axis=-1).argmin(axis=-1)
    for i, cls in enumerate(classes):
        mask[mask_idx == i] = cls.code
    return np.stack([mask, mask, mask], axis=-1)


if __name__ == "__main__":

    # --- setup parameters ---
    cls = LumenStoneClasses.S2()
    in_folder = Path("/Users/xubiker/dev/LumenStone/S2_new/masks_colored/")
    out_folder = Path("/Users/xubiker/dev/LumenStone/S2_new/masks/")
    # --- end of setup parameters ---

    out_folder.mkdir(parents=True, exist_ok=True)
    mask_paths = [
        p for p in in_folder.iterdir() if p.is_file() and p.suffix == ".bmp"
    ]
    mask_paths = sorted(mask_paths)

    for p in tqdm(mask_paths):
        mask_colored = cv2.imread(str(p))[:, :, ::-1]
        mask = color_to_mask(mask_colored, cls)
        cv2.imwrite(str(out_folder / f"{p.stem}.png"), mask[:, :, ::-1])
