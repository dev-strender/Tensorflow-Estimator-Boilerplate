from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags    # pylint: disable=g-bad-import-order

from ._conventions import help_wrap


def define_model_and_data(model=None, dataset=None):
    """Register flags for specifying model & data to use.
    Args:
        model: model name.
        dataset: dataset name.
    Returns:
        A list of flags for core.py to marks as key flags.
    """

    key_flags = []
    if model:
        flags.DEFINE_string(
            name="model", default=None,
            help=help_wrap("Model name to use. It should be defined on models/"
                           "and referenced in supervisor.py "
                           ))

    if dataset:
        flags.DEFINE_string(
            name="dataset", default=None,
            help=help_wrap("Dataset name to use. You should give data_dir firt, "
                           "and parsing code shuld be in datasets/ and"
                           "referenced the dataset in supervisor.py"))

    return key_flags
