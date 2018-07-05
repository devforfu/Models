import os
from os.path import join, expanduser


def get_env_variable(name: str, default=None) -> str:
    """
    Gets environment variable if available.

    Args:
        name: An environment variable name.
        default: A fallback value if variable is not defined.

    Returns:
        value: Environment variable value or default

    Raises:
        RuntimeError: Variable is not defined and fallback value
            is not provided.

    """
    value = os.environ.get(name, default)
    if not value:
        raise RuntimeError(
            "Environment variable '%s' is not defined without "
            "fallback value provided" % name)
    return value


def path(arg, *args):
    args = [arg] + list(args)
    return expanduser(join(*args))


# -----------------------------------------------------------------------------
#                         Environ variables and paths
# -----------------------------------------------------------------------------


MODELS_FOLDER = get_env_variable(name='TF_MODELS_DIR', default='models')

LFPW_TRAIN = get_env_variable(
    name='LFPW_TRAIN',
    default=path('~', 'data', 'landmarks', 'lfpw', 'trainset'))

LFPW_VALID = get_env_variable(
    name='LFPW_TEST',
    default=path('~', 'data', 'landmarks', 'lfpw', 'testset'))
