import unittest

loader = unittest.TestLoader()
tests = loader.discover('.')
test_runner = unittest.runner.TextTestRunner()
test_runner.run(tests)
