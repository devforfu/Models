from os.path import join

from keras import backend as K
from keras.models import Model
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.layers import Input, Flatten
from keras.layers import Dense, Conv2D, Activation
from keras.layers import Dropout, BatchNormalization
from keras.layers import GlobalAvgPool2D, GlobalMaxPool2D
from keras.layers import ZeroPadding2D, MaxPooling2D, AveragePooling2D

from utils import path
from models import PretrainedModel
from config import NUM_OF_LANDMARKS
from generators import AnnotatedImagesGenerator
from basedir import LFPW_TRAIN, LFPW_VALID, MODELS_FOLDER


K.set_image_dim_ordering('tf')


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def main():
    input_shape = 250, 250, 3

    base = InceptionResNetV2(input_shape=input_shape, include_top=False)

    x = GlobalAvgPool2D()(base.output)
    x = Dense(units=500, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(units=500, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(units=500, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(units=NUM_OF_LANDMARKS*2, activation='linear')(x)

    model = Model(inputs=base.input, outputs=x)
    model.compile(optimizer='adam', loss=root_mean_squared_error)

    train_gen = AnnotatedImagesGenerator(
        root=LFPW_TRAIN,
        batch_size=32,
        target_size=input_shape[:2],
        infinite=True,
        model_preprocessing=preprocess_input,
        augment=True)

    valid_gen = AnnotatedImagesGenerator(
        root=LFPW_VALID,
        batch_size=32,
        target_size=input_shape[:2],
        infinite=True,
        model_preprocessing=preprocess_input,
        augment=False)

    model.fit_generator(train_gen,
                        steps_per_epoch=train_gen.n_batches,
                        validation_data=valid_gen,
                        validation_steps=valid_gen.n_batches)

    print('Done!')

    # model = PretrainedModel(input_shape=input_shape)
    # model.create(
    #     model_fn=lambda x: InceptionResNetV2(
    #         input_shape=x,
    #         weights='imagenet',
    #         include_top=False),
    #     prep_fn=preprocess_input, pool='avg')
    # model.compile(optimizer='adam')
    # model.create_model_folder(root=join(MODELS_FOLDER, 'face_landmarks'))
    #
    # callbacks = [
    #     CSVLogger(filename=model.history_path),
    #     EarlyStopping(patience=5, verbose=1),
    #     ModelCheckpoint(filepath=model.weights_path,
    #                     save_best_only=True,
    #                     save_weights_only=False)]
    #
    # model.train(
    #     train_folder=LFPW_TRAIN,
    #     valid_folder=LFPW_VALID,
    #     callbacks=callbacks,
    #     normalize=False)
    #
    # y_preds = model.predict_generator(LFPW_VALID)



if __name__ == '__main__':
    main()
