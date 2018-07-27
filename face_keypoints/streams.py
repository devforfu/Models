import math
from pathlib import Path
from itertools import islice

import numpy as np

from utils import imread, read_landmarks, crop, resize, glob_extensions


class AnnotatedImagesStream:
    """
    Yields batches of image and annotation files from the dataset root folder.
    """
    def __init__(self, root: str, batch_size: int=32, infinite: bool=False,
                 same_size_batches: bool=False, images_ext: str='jpg|jpeg|png',
                 points_ext: str='pts'):

        self.root = root
        self.batch_size = batch_size
        self.infinite = infinite
        self.same_size_batches = same_size_batches
        self.images_files = glob_extensions(self.root, images_ext)
        self.points_files = glob_extensions(self.root, points_ext)

        if not infinite and same_size_batches:
            raise ValueError('Incompatible configuration: cannot guarantee '
                             'same size of batches when yielding finite '
                             'number of files.')

        pairs = list(zip(self.images_files, self.points_files))
        for img_name, pts_name in pairs:
            assert Path(img_name).stem == Path(pts_name).stem

        n_files = len(pairs)
        if same_size_batches:
            n_batches = n_files // batch_size
        else:
            n_batches = int(math.ceil(n_files / batch_size))

        self._n_batches = n_batches
        self._iter = iter(pairs)
        self._count = 0

    @property
    def n_batches(self):
        return self._n_batches

    def __iter__(self):
        return self

    def __next__(self):
        if not self.infinite and self._count >= self._n_batches:
            raise StopIteration()
        item = self.next()
        self._count += 1
        return item

    def next(self):
        bs = self.batch_size
        if self.infinite and self._count == self._n_batches:
            self._iter = iter(list(zip(self.images_files, self.points_files)))
            self._count = 0
        x_batch, y_batch = zip(*[xy for xy in islice(self._iter, 0, bs)])
        return x_batch, y_batch


class FilesReader:
    """
    Takes batches with images with faces and their annotations, and converts
    them into Numpy arrays.
    """

    def __init__(self, target_size, padding):
        self.target_size = target_size
        self.padding = padding

    def __call__(self, x_batch, y_batch):
        images, landmarks = [], []
        for img_file, pts_file in zip(x_batch, y_batch):
            img = imread(img_file)
            xs, ys = read_landmarks(pts_file)
            pts = np.r_[xs, ys]
            cropped = crop(img, pts, padding=self.padding)
            new_img, new_pts = resize(*cropped, target_size=self.target_size)
            images.append(new_img)
            landmarks.append(new_pts)
        return np.array(images), np.array(landmarks)
