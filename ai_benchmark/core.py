import os
import logging
import numpy as np
import tensorflow as tf
from ai_benchmark import utils

logger = logging.getLogger('ai_benchmark')


class AIBenchmark:
    def __init__(self, use_cpu=None, verbose_level=1, seed=42):
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

        self.use_cpu = False
        if use_cpu:
            self.use_cpu = True

    def run(self, precision="normal", test_ids=None, training=True, inference=True, micro=False,
            cpu_cores=None, inter_threads=None, intra_threads=None):
        return utils.run_tests(
            training=training,
            inference=inference,
            micro=micro,
            verbose=self.verbose,
            use_cpu=self.use_cpu,
            precision=precision,
            _type="full",
            start_dir=self.cwd,
            test_ids=test_ids,
            cpu_cores=cpu_cores,
            inter_threads=inter_threads,
            intra_threads=intra_threads,
        )
