import sys
from collections import OrderedDict

from .base import Callback


class StreamLogger(Callback):

    def __init__(self, output=sys.stdout, formatter='default',
                 formatter_config=None, logging_frequency=1):

        super().__init__()
        self.output = output
        self.formatter = get_formatter(formatter, formatter_config)
        self.logging_frequency = logging_frequency
        self._counter = None

    def write(self, string):
        self.output.write(string)
        self.output.write('\n')
        self.output.flush()

    def on_training_start(self):
        self._counter = 0

    def on_training_end(self):
        self._counter = None

    def on_epoch_end(self, epoch, metrics):
        self._counter += 1
        if self._counter % self.logging_frequency == 0:
            self.write(self.formatter.to_string(metrics))


_formatters = {}


class MetricsFormatter:

    def __init__(self, stats_formats=None, aliases=None,
                 default_format='2.6f', suppress_metrics=None):

        self.stats_formats = stats_formats
        self.aliases = aliases
        self.default_format = default_format
        self.suppress_metrics = set(suppress_metrics or [])

    def to_string(self, metrics):
        if self.stats_formats is None:
            return str(metrics)

        format_strings = []
        for name, value in metrics.items():
            if name in self.suppress_metrics:
                continue
            fmt = self.stats_formats.get(name, self.default_format)
            short_name = self.aliases.get(name, name)
            format_strings.append('%s: {%s:%s}' % (short_name, name, fmt))

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

        if not self.aliases:
            self.aliases = {
                'learning_rate': 'lr',
                'val_accuracy': 'val_acc',
                'accuracy': 'acc'}


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
    config = config or {}
    return _formatters[name](**config)


_register_formatter(DefaultFormatter, 'default')
