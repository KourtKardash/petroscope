"""
Calibration module for petroscope package.

This module provides an interface to perform calibration of images
using reference images of a mirror or an OLED screen.

"""

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageFilter
from scipy.optimize import curve_fit
from tqdm import tqdm

from petroscope.utils import logger


class ImageCalibrator:
    def __init__(
        self,
        reference_mirror_path: str | Path = None,
        reference_screen_path: str | Path = None,
        correct_distortion: bool = False,
    ) -> None:
        # Ensure at least one reference image is provided
        if not reference_mirror_path and not reference_screen_path:
            raise ValueError(
                "At least one reference image path should be provided."
            )

        # Validate and convert paths
        self._ref_mirror_path = self._validate_path(reference_mirror_path)
        self._ref_screen_path = self._validate_path(reference_screen_path)

        # Ensure distortion correction is only used for screen calibration
        if correct_distortion and not self._ref_screen_path:
            raise ValueError(
                "Distortion correction is only available for "
                "calibration with screen image."
            )

        self.correct_distortion = correct_distortion

        # Load reference images
        self._lum_map_mirror = self._illumination_mirror(self._ref_mirror_path)
        self._lum_map_screen = self._illumination_screen(self._ref_screen_path)
        self._lum_map = self._illumination_final(
            self._lum_map_mirror, self._lum_map_screen
        )

        if correct_distortion:
            self._distortion = self._distortion_screen(self._ref_screen_path)

    @staticmethod
    def _validate_path(path: str | Path | None) -> Path | None:
        """Converts to Path and checks if the file exists."""
        if path is None:
            return None
        path = Path(path) if not isinstance(path, Path) else path
        if not path.is_file():
            raise FileNotFoundError(f"Reference image does not exist: {path}")
        return path

    def _illumination_mirror(
        self, ref_img_path: Path | None
    ) -> np.ndarray | None:
        """Loads and preprocesses the reference mirror image."""
        if ref_img_path is None:
            return None
        img = (
            Image.open(ref_img_path)
            .convert("L")
            .filter(ImageFilter.GaussianBlur(radius=25))
        )
        mirror = np.array(img, dtype=np.float32) / 255
        illumination_mask = mirror + (1 - np.max(mirror))
        return illumination_mask

    def _illumination_screen(
        self, ref_img_path: Path | None
    ) -> np.ndarray | None:
        if ref_img_path is None:
            return None
        img = Image.open(ref_img_path)
        img = np.array(img)
        red_channel = img[:, :, 0]
        green_channel = img[:, :, 1]
        blue_channel = img[:, :, 2]
        height, width = red_channel.shape

        x_min, x_max = int(width * 0.4), int(width * 0.6)
        y_min, y_max = int(height * 0.4), int(height * 0.6)

        min_intensities = [
            np.min(channel[x_min:x_max, y_min:y_max])
            for channel in [red_channel, green_channel, blue_channel]
        ]
        min_index = np.argmin(min_intensities)
        chosen_channel = [red_channel, green_channel, blue_channel][min_index]
        add_channel_1 = [green_channel, blue_channel, red_channel][min_index]
        add_channel_2 = [blue_channel, red_channel, green_channel][min_index]

        centroids, intensities = self._get_centroids(chosen_channel,
                                                     add_channel_1,
                                                     add_channel_2)
        x_data = centroids[:, 0]
        y_data = centroids[:, 1]

        initial_guess = (1, np.mean(x_data), np.mean(y_data), 1000, 1000)

        popt, _ = curve_fit(self.gaussian_2d,
                            (x_data, y_data),
                            intensities,
                            p0=initial_guess,
                            maxfev=10000)

        A_opt, x0_opt, y0_opt, sigma_x_opt, sigma_y_opt = popt

        height, width = img.shape[:2]
        x = np.linspace(0, width - 1, width)
        y = np.linspace(0, height - 1, height)
        X, Y = np.meshgrid(x, y)

        Z = self.gaussian_2d((X, Y), A_opt, x0_opt, y0_opt,
                             sigma_x_opt, sigma_y_opt)

        illumination_mask = Z + (1 - np.max(Z))
        return illumination_mask

    def _distortion_screen(self, ref_img_path: Path) -> np.ndarray:
        """Placeholder for screen distortion calibration."""
        raise NotImplementedError

    def _illumination_final(
        self,
        mirror_map: np.ndarray | None,
        screen_map: np.ndarray | None,
    ) -> np.ndarray:
        map_1ch = None
        if mirror_map is not None and screen_map is not None:
            map_1ch = (mirror_map + screen_map) / 2
        else:
            if mirror_map is not None:
                map_1ch = mirror_map
            if screen_map is not None:
                map_1ch = screen_map
        map_3ch = np.repeat(map_1ch[:, :, np.newaxis], 3, axis=2)
        return map_3ch

    def gaussian_2d(
        self,
        coords: tuple[np.ndarray, np.ndarray],
        A: float,
        x0: float,
        y0: float,
        sigma_x: float,
        sigma_y: float
    ) -> np.ndarray:
        X, Y = coords
        norm_x = (X - x0) ** 2 / (2 * sigma_x ** 2)
        norm_y = (Y - y0) ** 2 / (2 * sigma_y ** 2)
        return A * np.exp(-(norm_x + norm_y))

    def _get_binary_image(
        self,
        chosen_channel: np.ndarray,
        add_channel_1: np.ndarray,
        add_channel_2: np.ndarray,
        window_size: int = 128
    ) -> np.ndarray:
        proc_image = np.zeros_like(chosen_channel, dtype=np.uint8)
        height, width = chosen_channel.shape
        for y in range(0, height, window_size):
            for x in range(0, width, window_size):
                window = chosen_channel[y:y+window_size, x:x+window_size]
                add_window_1 = add_channel_1[y:y+window_size, x:x+window_size]
                add_window_2 = add_channel_2[y:y+window_size, x:x+window_size]

                threshold = np.mean(window) + np.std(window)
                threshold_1 = np.mean(add_window_1) + 2*np.std(add_window_1)
                threshold_2 = np.mean(add_window_2) + 2*np.std(add_window_2)

                binary_wnd = ((window > threshold) &
                              (add_window_1 < threshold_1) &
                              (add_window_2 < threshold_2))
                proc_image[y:y+window_size, x:x+window_size][binary_wnd] = 255
        return proc_image
 
    def _get_centroids(
        self,
        chosen_channel: np.ndarray,
        add_channel_1: np.ndarray,
        add_channel_2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        thresh = self._get_binary_image(chosen_channel,
                                        add_channel_1,
                                        add_channel_2)
        image = cv2.medianBlur(thresh, 9)

        contours, _ = cv2.findContours(image,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        areas = np.array([cv2.contourArea(contour) for contour in contours])
        mean_number = np.mean(areas) - 1.5 * np.std(areas)
        filtered_contours = [contour for contour, area in zip(contours, areas)
                             if area >= mean_number]

        centroids = []
        intensities = []

        for contour in filtered_contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
                centroids.append((cX, cY))

                mask = np.zeros_like(image)
                cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)
                mean_intensity = cv2.mean(chosen_channel / 255, mask=mask)[0]
                intensities.append(mean_intensity)

        centroids = np.array(centroids)
        intensities = np.array(intensities)

        return centroids, intensities

    def calibrate(
        self, img_path: Path, out_path: Path, quiet: bool = False
    ) -> None:
        try:
            img = np.array(Image.open(img_path)).astype(np.float32) / 255

            img_corrected = img / self._lum_map
            img_corrected = np.clip(img_corrected, 0, 1)

            if self.correct_distortion:
                # Placeholder for distortion correction
                img_corrected = img_corrected

            img_res = Image.fromarray((img_corrected * 255).astype(np.uint8))
            img_res.save(out_path, quality=95)
            if not quiet:
                logger.info(f"Saved calibrated image to {out_path}")
        except Exception as e:
            logger.error(f"Error during calibration of {img_path}: {e}")

    def calibrate_batch(
        self,
        src: Iterable[Path] | Path,
        out: Path,
        quiet: bool = False,
    ) -> None:
        try:
            if isinstance(src, Path):
                if src.is_file():
                    src = [src]
                else:
                    if not src.is_dir():
                        raise ValueError(f"Input folder does not exist: {src}")
                    src = src.iterdir()
            src = [
                p
                for p in src
                if p.is_file() and p.suffix in (".jpg", ".png", ".bmp")
            ]
            if not src:
                logger.warning("No images found in the input folder.")
                return

            out.mkdir(parents=True, exist_ok=True)
            if not quiet:
                src = tqdm(src, desc="Calibrating images")

            for img_p in src:
                self.calibrate(img_p, out / img_p.name, quiet=True)
        except Exception as e:
            logger.error(f"Error during batch calibration: {e}")
