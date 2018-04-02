import os

COMPETITION_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.path.expanduser(
    '~/data/kaggle/dog-breed-identification/full')
TRAIN_IMAGES = os.path.join(DATA_ROOT, 'train')
VALID_IMAGES = os.path.join(DATA_ROOT, 'valid')
TEST_IMAGES = os.path.join(DATA_ROOT, 'test')
LABELS_FILE = os.path.join(DATA_ROOT, 'labels.csv')
