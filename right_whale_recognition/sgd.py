import os
from pathlib import Path

import numpy as np
from tpot import TPOTClassifier
from sklearn.metrics import log_loss
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import RandomizedSearchCV
from swissknife.kaggle.datasets import KaggleClassifiedImagesSource

from basedir import TRAIN_DATA, VALID_DATA, EXTENDED_LABELS


def create_targets(source, folder):
    """Converts file paths into target labels."""

    labels = [
        one_hot.argmax()
        for one_hot in (
            source.identifier_to_label[Path(filename).stem]
            for filename in os.listdir(folder))]
    return np.asarray(labels)


def main():
    source = KaggleClassifiedImagesSource(labels_path=EXTENDED_LABELS,
                                          label_column='whaleID',
                                          id_column='Image')

    x_train = np.load('inception_train_features.npy')
    y_train = create_targets(source, TRAIN_DATA)

    x_valid = np.load('inception_valid_features.npy')
    y_valid = create_targets(source, VALID_DATA)

    model = TPOTClassifier(generations=5,
                           population_size=10,
                           verbosity=2, n_jobs=-1)

    model.fit(x_train, y_train)
    classes = sorted(list(set(y_train)))

    probs = model.predict_proba(x_valid)
    preds = np.argmax(probs, axis=1)
    loss = log_loss(y_valid, preds, labels=classes)
    accuracy = np.mean(preds == y_valid)
    print(f'Test loss: {loss:2.6f} - accuracy: {accuracy:2.2%}')

    model.export('tpot_discovered_model.py')


if __name__ == '__main__':
    main()
