import numpy as np

from utils import to_grayscale
from streams import AnnotatedImagesStream, FilesReader
from augmentation import AugmentationsList
from augmentation import Rotation, Shift, HorizontalFlip, GammaCorrection


class AnnotatedImagesGenerator:

    def __init__(
            self,
            root: str,
            padding: int = 20,
            batch_size: int = 16,
            target_size: tuple = (120, 120),
            infinite: bool = True,
            same_size_batches: bool = True,
            augment: bool = True,
            grayscale: bool = False,
            shift_x: bool = True,
            shift_y: bool = True,
            shift_range: tuple = (-5, 5),
            rotation_range: tuple = (-10, 10),
            gamma_range: tuple = (0.5, 1.5),
            default_probability: float = 0.5,
            normalize: bool = True,
            probabilities: dict = None,
            model_preprocessing=None):

        stream = AnnotatedImagesStream(
            root=root,
            batch_size=batch_size,
            infinite=infinite,
            same_size_batches=same_size_batches)

        reader = FilesReader(
            target_size=target_size,
            padding=padding)

        probs = probabilities or {}
        p = default_probability

        if augment:
            cx, cy = target_size[0] / 2, target_size[1] / 2
            rotation = Rotation(
                center=(cx, cy),
                angle_range=rotation_range,
                probability=probs.get('rotation', p))
            shift = Shift(
                shift_range=shift_range,
                shift_x=shift_x,
                shift_y=shift_y,
                probability=probs.get('shift', p))
            flip = HorizontalFlip(
                probability=probs.get('flip', p))
            gamma = GammaCorrection(
                gamma_range=gamma_range,
                probability=probs.get('gamma', p))
            augmentation = AugmentationsList([rotation, shift, flip, gamma])
        else:
            augmentation = None

        self.target_size = target_size
        self.stream = stream
        self.reader = reader
        self.grayscale = grayscale
        self.n_batches = stream.n_batches
        self.augment = augment
        self.normalize = normalize
        self.augmentation = augmentation
        self.model_preprocessing = model_preprocessing

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        xy_batch = self.reader(*next(self.stream))
        if self.grayscale:
            xy_batch = to_grayscale(*xy_batch)
        if self.augment:
            xy_batch = self.augmentation.apply_to_batch(*xy_batch)

        x_batch, y_batch = xy_batch
        if self.model_preprocessing:
            x_batch = self.model_preprocessing(x_batch.astype(float))

        if self.normalize:
            adjust = self.target_size[0] / 2
            if self.model_preprocessing is None:
                x_batch = x_batch.astype(np.float) / 255.
            y_batch = (y_batch - adjust) / adjust

        xy_batch = x_batch, y_batch
        return xy_batch
