import sys
from collections import OrderedDict

from .base import Callback


class StreamLogger(Callback):

    def __init__(self, output=sys.stdout, formatter='default',
                 format_config=None):

        super().__init__()
        self.output = output
        self.formatter = get_formatter(formatter, format_config)

    def write(self, string):
        self.output.write(string)
        self.output.write('\n')
        self.output.flush()

    def on_training_start(self):
        self.write('Model training started')

    def on_training_end(self):
        self.write('Model training ended')

    def on_epoch_end(self, epoch, metrics):
        self.write(self.formatter.to_string(metrics))


_formatters = {}


class MetricsFormatter:

    def __init__(self, stats_formats=None):
        self.stats_formats = stats_formats

    def to_string(self, metrics):
        if self.stats_formats is None:
            return str(metrics)
        format_strings = [
            '%s: {%s:%s}' % (name, name, value)
            for name, value in self.stats_formats.items()]
        format_string = ' - '.join(format_strings)
        return format_string.format(**metrics)


class DefaultFormatter(MetricsFormatter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.stats_formats:
            stats_formats = OrderedDict()
            stats_formats['epoch'] = '05d'
            stats_formats['loss'] = '2.6f'
            stats_formats['val_loss'] = '2.6f'
            stats_formats['accuracy'] = '2.2%'
            stats_formats['val_accuracy'] = '2.2%'
            self.stats_formats = stats_formats


def _register_formatter(cls, alias=None):
    global _formatters
    name = alias or cls.__name__
    _formatters[name] = cls


def get_formatter(name, config=None) -> MetricsFormatter:
    if not isinstance(name, str):
        return name
    if name not in _formatters:
        raise ValueError(
            'unexpected formatter name, available formatters are: %s' %
            sorted(list(_formatters.keys())))
    return _formatters[name](config or {})


_register_formatter(DefaultFormatter, 'default')
