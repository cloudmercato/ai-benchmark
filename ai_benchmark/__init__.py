# -*- coding: utf-8 -*-
# Copyright 2019-2020 by Andrey Ignatov. All Rights Reserved.

import os
import logging
import numpy as np
import tensorflow as tf
from ai_benchmark import utils

handler = logging.StreamHandler()
logger = logging.getLogger('ai_benchmark')
logger.addHandler(handler)

VERSION = (0, 1, 2, 'cm')
__version__ = '.'.join([str(i) for i in VERSION])


class AIBenchmark:

    def __init__(self, use_CPU=None, verbose_level=1, seed=42):

        self.tf_ver_2 = utils.parse_version(tf.__version__) > utils.parse_version('1.99')
        self.verbose = verbose_level
        logger.setLevel(30 - self.verbose*10)

        utils.print_intro()

        np.warnings.filterwarnings('ignore')

        try:

            if verbose_level < 3:

                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

                if self.tf_ver_2:
                    tf_logger = tf.get_logger()
                    tf_logger.disabled = True
                    tf_logger.setLevel(logging.ERROR)

                elif utils.parse_version(tf.__version__) > utils.parse_version('1.13'):
                    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

                else:
                    tf.logging.set_verbosity(tf.logging.ERROR)

            else:

                if self.tf_ver_2:
                    tf_logger = tf.get_logger()
                    tf_logger.disabled = True
                    tf_logger.setLevel(logging.INFO)

                elif utils.parse_version(tf.__version__) > utils.parse_version('1.13'):
                    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

                else:
                    tf.logging.set_verbosity(tf.logging.INFO)

        except Exception as err:
            logger.warning("%s", err)

        np.random.seed(seed)
        self.cwd = os.path.dirname(__file__)

        self.use_CPU = False
        if use_CPU:
            self.use_CPU = True

    def run(self, precision="normal", test_ids=None):
        return utils.run_tests(
            training=True, inference=True, micro=False, verbose=self.verbose,
            use_CPU=self.use_CPU, precision=precision, _type="full", start_dir=self.cwd,
            test_ids=test_ids,
        )

    def run_inference(self, precision="normal", test_ids=None):
        return utils.run_tests(
            training=False, inference=True, micro=False, verbose=self.verbose,
            use_CPU=self.use_CPU, precision=precision, _type="inference", start_dir=self.cwd,
            test_ids=test_ids,
        )

    def run_training(self, precision="normal", test_ids=None):
        return utils.run_tests(
            training=True, inference=False, micro=False, verbose=self.verbose,
            use_CPU=self.use_CPU, precision=precision, _type="training", start_dir=self.cwd,
            test_ids=test_ids,
        )

    def run_micro(self, precision="normal", test_ids=None):
        return utils.run_tests(
            training=False, inference=False, micro=True, verbose=self.verbose,
            use_CPU=self.use_CPU, precision=precision, _type="micro", start_dir=self.cwd,
            test_ids=test_ids,
        )


if __name__ == "__main__":
    benchmark = AIBenchmark(use_CPU=None, verbose_level=1)
    results = benchmark.run(precision="normal")
