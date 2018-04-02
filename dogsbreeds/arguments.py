import argparse
from os.path import join


class TrainingArgsParser(argparse.ArgumentParser):
    """Commonly-used command line arguments tuning training or model selection
    process.
    """

    def __init__(self, competition_dir: str):
        super().__init__()
        self.competition_dir = competition_dir
        self.parsed = {}

        self.add_argument(
            '--model-name', type=str, default='default',
            help='Name of instantiated model')

        self.add_argument(
            '--n-epochs', type=int, default=100,
            help='Number of training epochs')

        self.add_argument(
            '--batch-size', type=int, default=128,
            help='Number of samples per training batch')

        self.add_argument(
            '--lr', type=float, default=0.001,
            help='Model training initial learning rate')

        self.add_argument(
            '--early-stopping', action='store_true',
            help='If set, then early stopping callback will be '
                 'enabled during model training')

        self.add_argument(
            '--early-stopping-patience', default=10, type=int,
            help='Number of epochs to wait for loss improvement before '
                 'stopping training process')

        self.add_argument(
            '--reduce-lr', action='store_true',
            help='If set, then learning rate reducing callback will be '
                 'enabled during model training')

        self.add_argument(
            '--reduce-lr-patience', default=5, type=int,
            help='Number of epochs to wait for loss improvement before '
                 'stopping training process')

        train_dir = join(competition_dir, 'train')
        self.add_argument(
            '--train', type=str, default=train_dir,
            help='Path to folder with training data')

        valid_dir = join(competition_dir, 'valid')
        self.add_argument(
            '--valid', type=str, default=valid_dir,
            help='Path to folder with validation data')

        test_dir = join(competition_dir, 'test')
        self.add_argument(
            '--test', type=str, default=test_dir,
            help='Path to folder with test data. This dataset is expected '
                 'to be unlabelled')

        labels_file = join(competition_dir, 'labels.csv')
        self.add_argument(
            '--labels', type=str, default=labels_file,
            help='Path to file with labels')

        self.add_argument(
            '-y', '--yes', action='store_true',
            help='If provided, then answer Y to all questions')

    def parse(self):
        self.parsed = vars(self.parse_args())

    def __getattr__(self, item):
        parsed = self.__dict__['parsed']
        if item not in parsed:
            raise AttributeError(item)
        return parsed[item]


class DogsBreedsArgsParser(TrainingArgsParser):

    def __init__(self, competition_dir):
        super().__init__(competition_dir)
        self.add_argument(
            '--preprocessing', choices=['model', 'data'], default='data',
            help='Preprocess samples using model preprocessing function or'
                 'using feature-wise normalization of data instead')
