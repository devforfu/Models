import os
import re
import math
from pathlib import Path
from itertools import chain
from os.path import join, expanduser, expandvars, abspath

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def calculate_layout(num_axes, n_rows=None, n_cols=None):
    """Calculates number of rows/columns required to fit `num_axes` plots
    onto figure if specific number of columns/rows is specified.
    """
    if n_rows is not None and n_cols is not None:
        raise ValueError(
            'cannot derive number of rows/columns if both values provided')
    if n_rows is None and n_cols is None:
        n_cols = 2
    if n_rows is None:
        n_rows = max(1, math.ceil(num_axes / n_cols))
    else:
        n_cols = max(1, math.ceil(num_axes / n_rows))
    return n_rows, n_cols


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


def best_checkpoint_path(root):
    def parse_loss(string):
        try:
            [loss] = re.findall('_([\d]+\.[\d]+).hdf5$', string)
            return float(loss)
        except ValueError:
            return None

    checkpoints = [int(ts) for ts in os.listdir(root)]
    most_recent = join(root, str(checkpoints[np.argmax(checkpoints)]))

    best_loss, best_file = np.inf, None

    for filename in os.listdir(most_recent):
        if not filename.endswith('.hdf5'):
            continue
        value = parse_loss(filename)
        if value is None:
            continue
        if value < best_loss:
            best_loss = value
            best_file = join(most_recent, filename)

    return best_file


def path(part, *parts):
    return abspath(expandvars(expanduser(join(part, *parts))))


def show_images(images, pts_pred, pts_true=None, n_cols=4, cmap=None):
    n_rows, n_cols = calculate_layout(len(images), n_cols=n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
    axes = axes.flatten()

    for ax in axes:
        ax.set_axis_off()

    for i, (img, pts, ax) in enumerate(zip(images, pts_pred, axes)):
        xs, ys = split_xy(pts)
        if img.shape[-1] == 1:
            img = img.reshape(img.shape[0], img.shape[1])
            cmap = 'gray'
        ax.imshow(img, cmap=cmap)
        ax.set_title(img.shape)
        ax.scatter(xs, ys, color='darkorange', edgecolor='white', s=20)
        if pts_true is not None:
            ax.scatter(*split_xy(pts_true[i]),
                       color='royalblue', edgecolor='white', alpha=0.75, s=15)
