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
            '-C', '--cpu-cores', default=None, type=int,
            help='Number of CPU cores to use.',
        )
        self.add_argument(
            '-b', '--intra-threads', default=None, type=int,
            help='inter_op_parallelism_threads'
        )
        self.add_argument(
            '-B', '--inter-threads', default=None, type=int,
            help='intra_op_parallelism_threads'
        )
        self.add_argument(
            '-T', '--run-training', default=True, type=int, choices=(0, 1),
            help='Run training benchmark',
        )
        self.add_argument(
            '-i', '--run-inference', default=True, type=int, choices=(0, 1),
            help='Run inference benchmark',
        )
        self.add_argument(
            '-m', '--run-micro', default=False, type=int, choices=(0, 1),
            help='Run micro benchmark',
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
        use_cpu=parsed_args.use_cpu,
        verbose_level=parsed_args.verbose,
        seed=parsed_args.seed,
    )
    test_info, results = benchmark.run(
        precision=parsed_args.precision,
        test_ids=parsed_args.test_ids,
        training=parsed_args.run_training,
        inference=parsed_args.run_inference,
        micro=parsed_args.run_micro,
        cpu_cores=parsed_args.cpu_cores,
        inter_threads=parsed_args.inter_threads,
        intra_threads=parsed_args.intra_threads,
    )
    if parsed_args.json:
        output = vars(results)
        output['test_results'] = {
            k: vars(v)
            for k, v in output['test_results'].items()
        }
        output['test_info'] = vars(test_info)
        output['test_info'].pop('results', None)
        print(json.dumps(output, indent=4))


if __name__ == '__main__':
    main()
