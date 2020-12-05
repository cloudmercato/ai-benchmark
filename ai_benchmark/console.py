#!/usr/bin/env python
import argparse


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
            '-p', '--precision', default='normal', type=str, choices=('normal', 'high'),
            help='normal or high, if high is selected, the benchmark will execute 10 times more runs for each test.'
        )


parser = MainArgumentParser()


def main():
    """Main runner for shell"""
    from ai_benchmark import AIBenchmark

    parsed_args = parser.parse_known_args()[0]

    benchmark = AIBenchmark(
        use_CPU=parsed_args.use_cpu,
        verbose_level=parsed_args.verbose,
    )
    results = benchmark.run(
        precision=parsed_args.precision
    )


if __name__ == '__main__':
    main()
