#  Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

import datetime
from flags import core as flags_core
from hooks import hooks_helper
from utils import distribution_utils

_DATE_TIME_FORMAT_PATTERN = "%Y-%m-%d-%H_%M_%S"


def define_flags():
    flags_core.define_base()
    flags_core.define_performance(num_parallel_calls=False)
    flags_core.define_model_and_data(model='Simple', dataset='CIFAR10')
    flags.adopt_module_key_flags(flags_core)
    flags_core.set_defaults(data_dir='data/cifar10/',
                            # default checkpoints are stored with time-based directory,
                            # but if you want you could change directory naming rule
                            # for your convenience for tracking experiments
                            model_dir='checkpoints/%s/' %
                                      datetime.datetime.now().strftime(_DATE_TIME_FORMAT_PATTERN),
                            batch_size=16,
                            train_epochs=40)


class Supervisor(object):
    def select_model(self, flags_obj):
        model_name = flags_obj.model
        if model_name == 'Simple':
            from models.simple_net import SimpleNet
            return SimpleNet()
        else:
            tf.logging.warn('Unimplemented model OR wrong parameter')

    def select_dataset(self, flags_obj):
        dataset_name = flags_obj.dataset
        if dataset_name == 'CIFAR10':
            from datasets.cifar10 import CIFAR10
            return CIFAR10(flags_obj)
        else:

            tf.logging.warn('Not available dataset')

    @staticmethod
    def thisstatic():
        pass

    def train_and_evaluate(self, flags_obj):
        """Run CIFAR10 training and eval loop.
            Args:
                flags_obj: An object containing parsed flag values.
            """
        session_config = tf.ConfigProto(
            inter_op_parallelism_threads=flags_obj.inter_op_parallelism_threads,
            intra_op_parallelism_threads=flags_obj.intra_op_parallelism_threads,
            allow_soft_placement=True)

        distribution_strategy = distribution_utils.get_distribution_strategy(
            flags_core.get_num_gpus(flags_obj), flags_obj.all_reduce_alg)

        run_config = tf.estimator.RunConfig(
            train_distribute=distribution_strategy, session_config=session_config)

        model = self.select_model(flags_obj)
        model_function = model.model_fn

        cifar10_classifier = tf.estimator.Estimator(
            model_fn=model_function,
            model_dir=flags_obj.model_dir,
            config=run_config
        )

        dataset = self.select_dataset(flags_obj)

        # Set up hook that outputs training logs every 100 steps.
        train_hooks = hooks_helper.get_train_hooks(
            flags_obj.hooks, model_dir=flags_obj.model_dir,
            batch_size=flags_obj.batch_size)

        # Train and evaluate model.
        for _ in range(flags_obj.train_epochs // flags_obj.epochs_between_evals):
            cifar10_classifier.train(input_fn=lambda: dataset.get_train_set(), hooks=train_hooks)
            eval_results = cifar10_classifier.evaluate(input_fn=lambda: dataset.get_validation_set())
            print('\nValidation results:\n\t%s\n' % eval_results)


def main(argv):
    sv = Supervisor()
    mode = argv[1]
    if mode == 'train':
        pass
    elif mode == 'eval':
        pass
    elif mode == 'train_and_eval':
        sv.train_and_evaluate(flags.FLAGS)
    elif mode == 'export':
        pass
    else:
        tf.logging.warn('You should choose mode in [{}, {}, {}, {}]'.format(
            'train', 'eval', 'train_and_eval', 'export'
        ))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    define_flags()
    absl_app.run(main)