import cv2 as cv
import numpy as np


class RandomAugmentationMixin:

    def __init__(self, probability=0.5):
        self.probability = probability

    def apply(self, image, landmarks):
        if self.probability >= np.random.rand():
            new_image, new_landmarks = image.copy(), landmarks.copy()
            return self._apply(new_image, new_landmarks)
        else:
            return image, landmarks

    def _apply(self, image, landmarks):
        raise NotImplementedError()


class AffineTransformation(RandomAugmentationMixin):

    def __init__(self, matrix, **kwargs):
        super().__init__(**kwargs)
        assert matrix.shape == (2, 3)
        self.matrix = matrix

    @property
    def rotation(self):
        return self.matrix[:2, :2]

    @property
    def shift(self):
        return self.matrix[:, -1].reshape(-1, 1)

    def _apply(self, image, landmarks):
        cols, rows = image.shape[:2]
        new_image = cv.warpAffine(image, self.matrix, (rows, cols))
        n = len(landmarks) // 2
        xy = landmarks.reshape(-1, n)
        xy_transformed = self.rotation @ xy + self.shift
        new_landmarks = xy_transformed.flatten()
        return new_image, new_landmarks


class Rotation(AffineTransformation):

    def __init__(self, center, angle_range=(-10, 10), **kwargs):
        angle = np.random.uniform(*angle_range)
        matrix = cv.getRotationMatrix2D(center, angle, 1)
        super().__init__(matrix, **kwargs)


class Shift(AffineTransformation):

    def __init__(self, shift_range=(-5, 5), shift_x=True, shift_y=True,
                 **kwargs):

        shifts = np.array([0, 0])
        if shift_x:
            shifts[0] = np.random.uniform(*shift_range)
        if shift_y:
            shifts[1] = np.random.uniform(*shift_range)
        rotation = np.eye(2)
        matrix = np.c_[rotation, shifts.reshape(-1, 1)]
        super().__init__(matrix, **kwargs)


class HorizontalFlip(RandomAugmentationMixin):

    def _apply(self, image, landmarks):
        new_image = cv.flip(image, 1)
        new_landmarks = landmarks.copy()
        last = len(landmarks)
        half = last // 2
        n = new_image.shape[0]
        new_landmarks[0:half] = (n - 1) - landmarks[0:half]
        new_landmarks[half:last] = landmarks[half:last]
        return new_image, new_landmarks


class GammaCorrection(RandomAugmentationMixin):

    def __init__(self, gamma_range=(0.5, 1.5), **kwargs):
        super().__init__(**kwargs)
        min_gamma, max_gamma = gamma_range
        delta = max_gamma - min_gamma
        self.gamma = delta * np.random.rand() + min_gamma

    def _apply(self, image, landmarks):
        inv_gamma = 1.0 / self.gamma
        table = np.array([
            255 * ((i / 255.0) ** inv_gamma)
            for i in np.arange(0, 256)])
        new_image = cv.LUT(image.astype(np.uint8), table.astype(np.uint8))
        return new_image, landmarks


class AugmentationsList:

    def __init__(self, augmentations):
        self.augmentations = augmentations
        for aug in self.augmentations:
            assert hasattr(aug, 'apply')

    def apply(self, image, landmarks):
        old_shape = image.shape
        sample = image, landmarks
        for aug in self.augmentations:
            sample = aug.apply(*sample)
        new_image, new_landmarks = sample
        return new_image.reshape(old_shape), new_landmarks

    def apply_to_batch(self, x_batch, y_batch):
        new_x_batch = np.zeros_like(x_batch)
        new_y_batch = np.zeros_like(y_batch)

        for i, (x, y) in enumerate(zip(x_batch, y_batch)):
            new_x, new_y = self.apply(x, y)
            new_x_batch[i] = new_x
            new_y_batch[i] = new_y

        return new_x_batch, new_y_batch
