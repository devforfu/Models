import argparse


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-shape', '-i', required=True, type=valid_shape,
        help='[default: %(default)s] The shape of input tensor '
             '(channels-last order is expected)',
        metavar='<SHAPE>'
    )
    parser.add_argument(
        '--optimizer', '-opt', type=valid_optimizer, default='adam',
        help='[default: %(default)s] The model training optimizer.',
        metavar='<OPT>'
    )
    parser.add_argument(
        '--learning-rate', '-lr', type=float, default=0.01,
        help='[default: %(default)s] The optimizer\'s learning rate.',
        metavar='<LR>'
    )
    parser.add_argument(
        '--n-epochs', '-n', default=100, type=int,
        help='[default: %(default)s] The number of training epochs',
        metavar='<NE>'
    )
    parser.add_argument(
        '--pool', '-pl', default='avg', choices=['flatten', 'avg', 'max'],
        help='[default: %(default)s] The pooling layer placed before dense '
             'layers. The layers except "flatten" allow to train models on '
             'input tensors of various size.',
        metavar='<PL>'
    )
    parser.add_argument(
        '--n-dense', '-nd', default=5, type=int,
        help='[default: %(default)s] The number of top dense layers. These '
             'layers are put on top of the bottleneck model extracting '
             'features to predict landmarks.',
        metavar='<ND>'
    )
    parser.add_argument(
        '--units', '-us', type=valid_units, default=500,
        help='The number of units per dense layer. Should be an integer to '
             'use the same value for all layers, or a list of values of the '
             'length equal to the number of dense layers.',
        metavar='<US>'
    )
    parser.add_argument(
        '--patience', '-pt', default=0.1, type=valid_patience,
        help='[default: %(default)s] patience value for EarlyStopping callback. '
             'If integer provided, then this value is interpreted as number of '
             'epochs before without loss improvement before stopping. If '
             'floating point number, then the patience level is defined as a '
             'fraction of training epochs.',
        metavar='<PT>'
    )
    parser.add_argument(
        '--batch-norm', '-bn', action='store_true',
        help='Use batch normalization in all available layers.'
    )
    parser.add_argument(
        '--dropouts', '-dr', type=valid_dropouts,
        help='The dropouts layers probabilities for dense layers. If not '
             'provided, then the layers are not included into model. The '
             'number of provided dropouts should be equal to number of dense '
             'layers, or a single number if the same value should be used for '
             'all of them.',
        metavar='<DR>'
    )
    parser.add_argument(
        '--maxnorm', '-mn', default=None, type=float,
        help='[default: %(default)s] The maxnorm constraint applying to the '
             'network layers. If the model is a pretrained model, then this '
             'constraint is applied to the top layers only. Otherwise, all the '
             'layers are constrained with the same value.',
        metavar='<MN>'
    )
    parser.add_argument(
        '--l2-reg', '-l2', default=None, type=float,
        help='[default: %(default)s] The amount of L2 norm regularization '
             'applied to network layers. If the model is a pretrained model, '
             'then the regularization is applied to the top layers only. '
             'Otherwise, all the layers are constrained with the same value. '
             'If None, then the regularization is not applied.',
        metavar='<L2>'
    )
    parser.add_argument(
        '--identifier', '-id', default=None, type=str,
        help='[default: %(default)s] The unique identifier of the trained '
             'model. The identifier is only required for further analysis and '
             'doesn\'t affect training process. If provided then the '
             'identifier is saved into model folder as a text file.'
    )

    parsed = parser.parse_args(args)

    if isinstance(parsed.patience, float):
        parsed.patience = int(parsed.n_epochs * parsed.patience)

    return parsed


def valid_shape(value):
    try:
        shape = tuple([int(x) for x in value.split(',')])
        if len(shape) == 2:
            return shape + (1,)
        elif len(shape) == 3:
            return shape
        else:
            raise argparse.ArgumentTypeError('invalid shape: %s' % value)
    except ValueError:
        raise argparse.ArgumentTypeError('cannot parse tuple: %s' % value)


def valid_units(value):
    try:
        return [int(x) for x in value.split(',')]
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError('invalid units list: %s' % value)


def valid_patience(value):
    try:
        return int(value)
    except ValueError:
        try:
            patience = float(value)
            if 0 < patience <= 1:
                return patience
            raise argparse.ArgumentTypeError('invalid patience range')
        except ValueError:
            raise argparse.ArgumentTypeError('invalid patience value')


def valid_dropouts(value):
    try:
        return [float(x) for x in value.split(',')]
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError('invalid dropouts list: %s' % value)


def valid_optimizer(value):
    from keras.optimizers import get
    try:
        _ = get(value)
    except ValueError:
        raise argparse.ArgumentTypeError('unknown optimizer: %s' % value)
    return value
