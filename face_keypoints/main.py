import sys
from os.path import join
from pprint import pprint

from keras import backend as K
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint

from pretrained import get
from cli import parse_args
from models import PretrainedModel
from basedir import LFPW_TRAIN, LFPW_VALID, MODELS_FOLDER


K.set_image_dim_ordering('tf')


def main():
    args = parse_args(args=sys.argv[1:])
    (
        architecture,
        input_shape,
        optimizer,
        learning_rate,
        n_epochs,
        pool,
        n_dense,
        units,
        patience,
        batch_norm,
        dropouts,
        maxnorm,
        l2_reg,
        patience
    ) = (
        args.architecture, args.input_shape, args.optimizer, args.learning_rate,
        args.n_epochs, args.pool, args.n_dense, args.units, args.patience,
        args.batch_norm, args.dropouts or [], args.maxnorm, args.l2_reg,
        args.patience
    )

    print('Model training with parameters:')
    pprint(vars(args))

    model = PretrainedModel(input_shape=input_shape)

    model.create(
        *get(architecture),
        n_dense=n_dense,
        pool=pool,
        units=units,
        bn=batch_norm,
        dropouts=dropouts,
        maxnorm=maxnorm,
        l2_reg=l2_reg)

    ok = model.create_model_folder(
        root=join(MODELS_FOLDER, 'face_landmarks'),
        subfolder=args.identifier)

    if not ok:
        sys.exit(1)

    model.save_parameters(model.parameters_path)
    model.compile(optimizer=optimizer)

    callbacks = [
        CSVLogger(filename=model.history_path),
        EarlyStopping(patience=patience, verbose=1),
        ModelCheckpoint(filepath=model.weights_path,
                        save_best_only=True,
                        save_weights_only=False)]

    model.train(
        n_epochs=n_epochs,
        train_folder=LFPW_TRAIN,
        valid_folder=LFPW_VALID,
        callbacks=callbacks)

    avg_rmse = model.score(LFPW_VALID)
    print(f'Trained model validation RMSE: {avg_rmse:2.4f}')
    print(f'The folder with results: {model.subfolder}')
    print(f'Training history file: {model.history_path}')


if __name__ == '__main__':
    main()
