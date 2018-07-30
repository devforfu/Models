from os.path import join

from keras import backend as K
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from models import PretrainedModel
from basedir import LFPW_TRAIN, LFPW_VALID, MODELS_FOLDER


K.set_image_dim_ordering('tf')


def main():
    input_shape = 250, 250, 3

    model = PretrainedModel(input_shape=input_shape)
    model.create(
        model_fn=lambda x: InceptionResNetV2(
            input_shape=x,
            weights='imagenet',
            include_top=False),
        prep_fn=preprocess_input, pool='avg')
    model.compile(optimizer='adam')
    model.create_model_folder(root=join(MODELS_FOLDER, 'face_landmarks'))

    callbacks = [
        CSVLogger(filename=model.history_path),
        EarlyStopping(patience=5, verbose=1),
        ModelCheckpoint(filepath=model.weights_path,
                        save_best_only=True,
                        save_weights_only=False)]

    model.train(
        train_folder=LFPW_TRAIN,
        valid_folder=LFPW_VALID,
        callbacks=callbacks,
        normalize=False)

    y_preds = model.predict_generator(LFPW_VALID)



if __name__ == '__main__':
    main()
