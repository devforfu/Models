import os
import logging
import warnings
from pathlib import Path
from os.path import exists

import numpy as np
from tqdm import tqdm
from swissknife.config import console_logger
from swissknife.kaggle.datasets import KaggleTestImagesIterator
from swissknife.kaggle.datasets import KaggleClassifiedImagesSource

from pretrained import MODELS, get_pretrained_model
from basedir import TRAIN_DATA, VALID_DATA, EXTENDED_LABELS, TEST_DATA


class FeaturesExtractor:
    """Runs pretrained model without top layers on dataset and saves generated
    bottleneck features onto disk.
    """
    def __init__(self, build_fn, preprocess_fn, source,
                 target_size=(299, 299, 3), batch_size=128,
                 pool='avg'):

        self.build_fn = build_fn
        self.preprocess_fn = preprocess_fn
        self.source = source
        self.target_size = target_size
        self.batch_size = batch_size
        self.model = self.build_fn(
            weights='imagenet', include_top=False, pooling=pool)

    def __call__(self, folder, filename):
        stream = self.source(
            folder=folder, target_size=self.target_size,
            batch_size=self.batch_size, infinite=False)

        preprocessed = []
        with tqdm(total=stream.steps_per_epoch) as bar:
            for x_batch, y_batch in stream:
                x_preprocessed = self.preprocess_fn(x_batch)
                batch = self.model.predict_on_batch(x_preprocessed)
                preprocessed.append(batch)
                bar.update(1)

        all_features = np.vstack(preprocessed)
        np.save(filename, all_features)
        return filename


def create_targets(source, folder):
    """Converts file paths into target labels."""

    labels = [
        one_hot.argmax()
        for one_hot in (
            source.identifier_to_label[Path(filename).stem]
            for filename in os.listdir(folder))]
    return np.asarray(labels)


def main():
    logging.getLogger('tensorflow').disabled = True
    warnings.filterwarnings('ignore', category=FutureWarning)

    source = KaggleClassifiedImagesSource(
        labels_path=EXTENDED_LABELS,
        label_column='whaleID',
        id_column='Image')

    target_size = 299, 299, 3
    suffix = '_'.join([str(x) for x in target_size])
    all_train_features, all_valid_features = [], []
    log = console_logger()

    log.info('Extracting labelled data features')
    for model in MODELS:
        train_path = f'{model}_train_features_{suffix}.npy'
        valid_path = f'{model}_valid_features_{suffix}.npy'
        if not (exists(train_path) and exists(valid_path)):
            log.info('Running feature extractor: %s', model)
            build_fn, preprocess_fn = get_pretrained_model(model)
            extractor = FeaturesExtractor(
                build_fn=build_fn,
                preprocess_fn=preprocess_fn,
                source=source,
                target_size=target_size,
                batch_size=128)
            extractor(TRAIN_DATA, train_path)
            extractor(VALID_DATA, valid_path)
        all_train_features.append(np.load(train_path))
        all_valid_features.append(np.load(valid_path))

    x_train = np.hstack(all_train_features)
    x_valid = np.hstack(all_valid_features)
    y_train = create_targets(source, TRAIN_DATA)
    y_valid = create_targets(source, VALID_DATA)

    all_test_features = []
    log.info('Extracting test data features')
    for model in MODELS:
        test_path = f'{model}_test_features_{suffix}.npy'
        if not exists(test_path):
            log.info('Running feature extractor: %s', model)
            build_fn, preprocess_fn = get_pretrained_model(model)
            source = KaggleTestImagesIterator(
                test_folder=TEST_DATA,
                target_size=target_size,
                batch_size=128,
                with_identifiers=False)
            model = build_fn(
                weights='imagenet',
                include_top=False,
                pooling='avg')
            preprocessed = []
            with tqdm(total=source.n_batches) as bar:
                for batch in source:
                    features = model.predict_on_batch(preprocess_fn(batch))
                    preprocessed.append(features)
                    bar.update(1)
            np.save(test_path, np.vstack(preprocessed))
        all_test_features.append(np.load(test_path))

    x_test = np.hstack(all_test_features)
    log.info('Concatenating features')
    dataset = dict(
        x_train=x_train, x_valid=x_valid,
        y_train=y_train, y_valid=y_valid,
        x_test=x_test)
    np.savez('all_features.npz', **dataset)
    log.info('All features datasets saved onto disk')


if __name__ == '__main__':
    main()
