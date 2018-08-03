import sys
from os.path import join
from pprint import pprint

from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint

from cli import parse_args
from models import ResNet34
from basedir import MODELS_FOLDER, LFPW_TRAIN, LFPW_VALID


def main():
    args = parse_args(args=sys.argv[1:])

    input_shape, n_epochs, n_top, units, pool, l2, patience = (
        args.input_shape, args.n_epochs, args.n_dense, args.units,
        args.pool, args.l2_reg, args.patience)

    print('Model training with parameters:')
    pprint(vars(args))

    model = ResNet34(input_shape=input_shape)
    model.create(pool='avg', n_top=n_top, units=units, l2_reg=l2)
    model.compile(optimizer=args.optimizer)

    ok = model.create_model_folder(
        root=join(MODELS_FOLDER, 'face_landmarks'),
        subfolder=args.identifier)

    if not ok:
        sys.exit(1)

    model.save_parameters(model.parameters_path)

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
