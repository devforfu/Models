from pathlib import Path
from itertools import chain

import cv2 as cv
import numpy as np


def imread(filename):
    """Reads an image into RGB array format."""
    return cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2RGB)


def read_landmarks(filename, one_based_index=True):
    """
    Reads file in PTS format into two arrays with x and y
    landmarks positions.

    The implementation is based on analogous Menpo library function.
    """
    with open(filename) as file:
        lines = [line.strip() for line in file]

    line = lines[0]
    while not line.startswith('{'):
        line = lines.pop(0)

    xs, ys = [], []
    for line in lines:
        if line.strip().startswith('}'):
            continue
        x, y = line.split()[:2]
        xs.append(x)
        ys.append(y)

    offset = 1 if one_based_index else 0
    xs = np.array(xs, dtype=np.float) - offset
    ys = np.array(ys, dtype=np.float) - offset
    return xs, ys


def crop(image, landmarks, padding=None):
    """
    Crops image to contain only a region of face with landmarks with
    optional padding region.
    """
    xs, ys = split_xy(landmarks.copy())
    bbox = [xs.min(), ys.min(), xs.max(), ys.max()]
    if padding is not None:
        bbox[0] -= padding
        bbox[1] -= padding
        bbox[2] += padding
        bbox[3] += padding
    left, top, right, bottom = [int(max(0, x)) for x in bbox]
    cropped = image[top:bottom, left:right, :]
    xs -= left
    ys -= top
    new_landmarks = np.r_[xs, ys]
    return cropped, new_landmarks


def split_xy(landmarks):
    n = len(landmarks) // 2
    return landmarks[:n], landmarks[n:]


def resize(image, landmarks, target_size):
    """
    Rescales image and its landmarks without keeping original aspect ratio.
    """
    image = np.copy(image)
    new_image = cv.resize(image, target_size)

    old_h, old_w = image.shape[:2]
    new_h, new_w = new_image.shape[:2]
    w_ratio = new_w / float(old_w)
    h_ratio = new_h / float(old_h)
    n = landmarks.shape[0] // 2

    new_landmarks = np.zeros_like(landmarks)
    for i in range(0, n):
        new_landmarks[i] = landmarks[i] * w_ratio
        new_landmarks[i + n] = landmarks[i + n] * h_ratio

    return new_image, new_landmarks


def glob_extensions(folder, extensions):
    folder = Path(folder)
    return sorted([
        path.as_posix()
        for path in chain(*[
            folder.glob('*.' + ext)
            for ext in parse_extensions(extensions)])])


def parse_extensions(string):
    return string.split('|') if '|' in string else [string]


def to_grayscale(x_batch, y_batch):
    """
    We ignore landmarks because their positions are not affected by
    converting image into another color space.
    """
    n, w, h = x_batch.shape[:3]
    grayscaled = np.zeros((n, w, h, 1))
    for i, image in enumerate(x_batch):
        new_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        grayscaled[i, :] = new_image.reshape((w, h, 1))
    return grayscaled, y_batch
