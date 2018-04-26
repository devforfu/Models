import os
from os.path import join, expanduser


COMPETITION_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.environ.get('KAGGLE_DATASETS', expanduser('~/data'))

# original competitions data
COMPETITION_DATA = join(DATA_ROOT, 'noaa-right-whale-recognition')
IMAGES_PATH = join(COMPETITION_DATA, 'imgs')
LABELS_PATH = join(COMPETITION_DATA, 'train.csv')

# data after separating into folders
PREPARED_DATA = join(COMPETITION_DATA, 'prepared')
TEST_DATA = join(PREPARED_DATA, 'test')
TRAIN_DATA = join(PREPARED_DATA, 'train')
VALID_DATA = join(PREPARED_DATA, 'valid')
EXTENDED_LABELS = join(PREPARED_DATA, 'labels.csv')

# mock data to validate custom algorithms
MNIST_IMAGES = expanduser('~/data/mnist')