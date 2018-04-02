"""
Main entry point to run data preprocessing and train models.

This module contains image transforming pipeline functions and common code
to parse CLI arguments, load different models architectures and save training
results onto disk.
"""
import warnings
from io import StringIO
from os.path import join
from contextlib import redirect_stdout

warnings.filterwarnings(action='ignore', category=FutureWarning)

from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from swissknife.files import SavingFolder
from swissknife.config import main_logger
from swissknife.transform import GeneratorPipeline
from swissknife.kaggle.datasets import KaggleClassifiedImagesSource

from models import get_model
from basedir import DATA_ROOT, TRAIN_IMAGES, VALID_IMAGES
from transforms import augment_images, shuffle_samples
from transforms import normalize_images, apply_to_samples
from arguments import DogsBreedsArgsParser


def main():
    args = DogsBreedsArgsParser(DATA_ROOT)
    args.parse()
    template = get_model(args.model_name)
    target_size = template.target_size
    model_name = '%s_bs%d_e%d' % (
        template.name, args.batch_size, args.n_epochs)

    log = main_logger(output_file=model_name + '.log')
    saver = SavingFolder(model_name, log=log)
    ok = saver.create_model_dir(ask_on_rewrite=(not args.yes))
    if not ok:
        log.info('Training cancelled')
        return

    checkpoint_name = '%s_{epoch:03d}_{val_loss:2.4f}.hdf5' % model_name
    checkpoint_path = join(saver.model_dir, checkpoint_name)

    callbacks = [
        ModelCheckpoint(filepath=checkpoint_path, save_best_only=True),
        CSVLogger(filename=saver.history_path)]

    if args.early_stopping:
        log.info('Early stopping callback enabled')
        callbacks.append(EarlyStopping(
            patience=args.early_stopping_patience,
            verbose=1))

    if args.reduce_lr:
        log.info('Learning rate reducing callback enabled')
        callbacks.append(ReduceLROnPlateau(
            patience=args.reduce_lr_patience,
            min_lr=1e-6,
            verbose=1))

    model = template.build(n_classes=120)
    K.set_value(model.optimizer.lr, args.lr)

    buf = StringIO()
    with redirect_stdout(buf):
        model.summary()
        template.layers_status(model)
        buf.seek(0)
    log.info('Model configuration:')
    for line in buf:
        log.info(line.strip())

    log.info('Network training parameters')
    params = args.parsed
    for key in sorted(params.keys()):
        value = params[key]
        log.info('.. %s: %s', key, value)

    image_loader = KaggleClassifiedImagesSource(
        labels_path=args.labels,
        label_column='breed')

    train_samples = image_loader(
        folder=TRAIN_IMAGES,
        target_size=target_size,
        batch_size=args.batch_size,
        infinite=True)

    valid_samples = image_loader(
        folder=VALID_IMAGES,
        target_size=target_size,
        batch_size=args.batch_size,
        infinite=True)

    def make_preprocessor(name):
        return {
            'data': normalize_images(target_size=target_size),
            'model': apply_to_samples(template.preprocessing_function)
        }[name]

    train_pipeline = GeneratorPipeline(
        train_samples,
        augment_images(horizontal_flip=True),
        shuffle_samples(),
        make_preprocessor(args.preprocessing))

    valid_pipeline = GeneratorPipeline(
        valid_samples,
        make_preprocessor(args.preprocessing))

    log.info('Start training model...')
    model.fit_generator(
        generator=train_pipeline,
        epochs=args.n_epochs,
        steps_per_epoch=train_samples.steps_per_epoch,
        validation_data=valid_pipeline,
        validation_steps=valid_samples.steps_per_epoch,
        callbacks=callbacks)

    log.info('Models training was finished!')


if __name__ == '__main__':
    main()
