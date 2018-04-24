MODELS = ['inception_v3', 'inception_resnet', 'xception']


def get_inception_v3():
    import keras.applications.inception_v3 as app
    return app.InceptionV3, app.preprocess_input


def get_inception_resnet():
    import keras.applications.inception_resnet_v2 as app
    return app.InceptionResNetV2, app.preprocess_input


def get_xception():
    import keras.applications.xception as app
    return app.Xception, app.preprocess_input


def get_pretrained_model(name):
    methods = {
        'inception_v3': get_inception_v3,
        'inception_resnet': get_inception_resnet,
        'xception': get_xception}
    getter = methods[name]
    return getter()
