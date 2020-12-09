import os
import unittest
from ai_benchmark import utils


class RunTestsTest(unittest.TestCase):
    @unittest.mock.patch('tensorflow.python.client.session.Session.run')
    def test_func(self, mock_run):
        utils.run_tests(
            training=True,
            inference=True,
            micro=False,
            verbose=0,
            use_cpu=None,
            precision='dry',
            _type="full",
            start_dir=os.path.dirname(__file__),
        )
        self.assertTrue(mock_run.called)
