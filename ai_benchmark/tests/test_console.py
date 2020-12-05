import unittest
from ai_benchmark import console


class MainTest(unittest.TestCase):
    # @unittest.mock.patch('sys.argv', [])
    @unittest.mock.patch('ai_benchmark.AIBenchmark.run')
    def test_func(self, mock_run):
        console.main()
        self.assertTrue(mock_run.called)
