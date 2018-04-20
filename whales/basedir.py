import os


COMPETITION_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.environ.get('KAGGLE_DATASETS', os.path.expanduser('~/data'))

ORIGINAL_DATA = os.path.join(DATA_ROOT, 'whale-categorization-playground')
PREPARED_DATA = os.path.join(ORIGINAL_DATA, 'prepared')
TRAIN_IMAGES = os.path.join(PREPARED_DATA, 'train')
VALID_IMAGES = os.path.join(PREPARED_DATA, 'valid')
TEST_IMAGES = os.path.join(ORIGINAL_DATA, 'test')
LABELS_FILE = os.path.join(ORIGINAL_DATA, 'train.csv')
