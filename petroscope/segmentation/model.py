from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


class GeoSegmModel(ABC):
    """
    An abstract class representing a generic model for geological segmentation.

    GeoSegmModel is an abstract class that serves as a base for all models that
    are used for geological segmentation. It provides abstract methods for
    initialization, loading the model, training and predicting images.

    Attributes:
        None

    Methods:
        initialize(self) -> None: An abstract method that initializes the model.

        load(self, saved_path: Path, **kwargs) -> None: An abstract method that
            loads the model from a given path.

        train(self, img_mask_paths: Iterable[tuple[Path, Path]], **kwargs) -> None:
            An abstract method that trains the model on a given set of images
            and masks.

        predict_image(self, image: np.ndarray) -> np.ndarray: An abstract method
            that predicts the segmentation of a given image.
    """

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def load(self, saved_path: Path, **kwargs) -> None:
        pass

    @abstractmethod
    def train(
        self, img_mask_paths: Iterable[tuple[Path, Path]], **kwargs
    ) -> None:
        pass

    @abstractmethod
    def predict_image(self, image: np.ndarray) -> np.ndarray:
        pass
