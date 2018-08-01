def get(name):
    keras_model_fn, prep_fn = {
        'inception_resnet_v2': inception_resnet_v2,
        'xception': xception,
        'inception_v3': inception_v3,
        'resnet50': resnet50
    }[name]()

    def partial(x):
        return keras_model_fn(
            input_shape=x, weights='imagenet', include_top=False)

    return partial, prep_fn


def inception_resnet_v2():
    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    from keras.applications.inception_resnet_v2 import preprocess_input
    return InceptionResNetV2, preprocess_input


def xception():
    from keras.applications.xception import Xception
    from keras.applications.xception import preprocess_input
    return Xception, preprocess_input


def inception_v3():
    from keras.applications.inception_v3 import InceptionV3
    from keras.applications.inception_v3 import preprocess_input
    return InceptionV3, preprocess_input


def resnet50():
    from keras.applications.resnet50 import ResNet50
    from keras.applications.resnet50 import preprocess_input
    return ResNet50, preprocess_input
