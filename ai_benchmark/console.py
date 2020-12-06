#!/usr/bin/env python
import argparse
import json
from ai_benchmark import config

TEST_IDS = [str(t.id) for t in config.BENCHMARK_TESTS]

class MainArgumentParser(argparse.ArgumentParser):
    """Parser with AI Benchmark arguments"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument(
            '-c', '--use-cpu', default=None, action='store_true',
            help='Run the tests on CPUs  (if tensorflow-gpu is installed)'
        )
        self.add_argument(
            '-v', '--verbose', default=1, type=int, choices=(0, 1, 2, 3),
            help='0: silent, 1: short summary, 2: more info, 3: TF logs'
        )
        self.add_argument(
            '-p', '--precision', default='normal', type=str, choices=('normal', 'high', 'dry'),
            help='normal or high, if high is selected, the benchmark will execute 10 times more runs for each test. dry do not run any iterations.'
        )
        self.add_argument(
            '-s', '--seed', default=42, type=int,
            help='Random seed',
        )
        self.add_argument(
            '-t', '--test-ids', default=None, nargs='+', choices=TEST_IDS,
            help="Select test by ID, all by default",
        )
        self.add_argument(
            '-j', '--json', action='store_true',
            help="Output results as JSON",
        )


parser = MainArgumentParser()


def main():
    """Main runner for shell"""
    from ai_benchmark import AIBenchmark

    parsed_args = parser.parse_known_args()[0]

    benchmark = AIBenchmark(
        use_CPU=parsed_args.use_cpu,
        verbose_level=parsed_args.verbose,
        seed=parsed_args.seed,
    )
    results = benchmark.run(
        precision=parsed_args.precision,
        test_ids=parsed_args.test_ids,
    )
    if parsed_args.json:
        output = vars(results)
        output['test_results'] = {
            k: vars(v)
            for k, v in output['test_results'].items()
        }
        print(json.dumps(output, indent=4))


if __name__ == '__main__':
    main()
