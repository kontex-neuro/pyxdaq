from pyxdaq.xdaq import get_XDAQ
from pathlib import Path
from pyxdaq.impedance import MeasurementStrategy, TestFrequency
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--runs', type=int, required=True, help='Number of runs to perform')
    parser.add_argument('--output', type=str, required=True, help='Output file')
    parser.add_argument(
        '--periods',
        type=int,
        help='Number of periods to measure (only one of --periods or --duration should be used)'
    )
    parser.add_argument(
        '--duration',
        type=float,
        help='Duration of measurement in seconds (only one of --periods or --duration should be used)'
    )
    parser.add_argument('--test_points', type=int, default=10, help='Number of test frequencies')
    parser.add_argument(
        '--low_frequency', type=float, default=30, help='Low frequency for test points'
    )
    parser.add_argument(
        '--high_frequency', type=float, default=7500, help='High frequency for test points'
    )
    parser.add_argument(
        '--channel',
        type=int,
        nargs='+',
        default=None,
        help='Specific channels to measure, if not provided all channels will be measured'
    )
    parser.add_argument(
        '--progress',
        type=int,
        default=1,
        help='Show progress bar, 0 for disabled, 1 or above for more detailed progress'
    )
    args = parser.parse_args()
    if args.periods is None and args.duration is None:
        parser.error('Either --periods or --duration must be used')
    if args.periods is not None and args.duration is not None:
        parser.error('Only one of --periods or --duration should be used')
    return args


xdaq = get_XDAQ(rhs=True)

if xdaq.numDataStream != 2:
    raise ValueError('This script is only compatible with one X3SR32 headstage')

args = get_args()
if args.periods is not None:
    measurement_strategy = MeasurementStrategy.from_periods(args.periods)
else:
    measurement_strategy = MeasurementStrategy.from_duration(args.duration)

test_frequencies = np.logspace(
    np.log10(args.low_frequency), np.log10(args.high_frequency), args.test_points
)
test_frequencies = np.array(
    [
        TestFrequency(f).get_actual(xdaq.sampleRate.rate, display_warning=False)
        for f in test_frequencies
    ]
)

if args.channel is not None:
    test_channels = np.array(args.channel)
else:
    test_channels = np.arange(16)

results = [
    np.array(
        [
            xdaq.measure_impedance(
                test_frequency=TestFrequency(frequency),
                channels=test_channels,
                progress=args.progress > 1,
                measurement_strategy=measurement_strategy,
                raw_measurement_only=True
            )
            for _ in tqdm(range(args.runs), desc='Runs', disable=args.progress == 0)
        ]
    )
    for frequency in tqdm(test_frequencies, desc='Measuring Impedance', disable=args.progress == 0)
]

np.savez_compressed(
    args.output,
    *results,
    test_frequencies=test_frequencies,
    test_channels=test_channels,
    sample_rate=[xdaq.sampleRate.rate],
    periods=[TestFrequency(f).get_period(xdaq.sampleRate.rate) for f in test_frequencies],
)
