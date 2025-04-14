from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from pyxdaq.impedance import Frequency, Strategy
from pyxdaq.xdaq import get_XDAQ


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--rhs', action='store_true')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs to perform')
    parser.add_argument('--output', type=str, default='res.npz', help='Output file')
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
        args.duration = 0.4
        print('Using default duration of 0.4 seconds')
    if args.periods is not None and args.duration is not None:
        parser.error('Only one of --periods or --duration should be used')
    return args


args = get_args()

xdaq = get_XDAQ(rhs=True)
print(xdaq.ports)

if xdaq.numDataStream > 2:
    raise ValueError('This script is only compatible with one X3SR32 headstage')

if args.periods is not None:
    strategy = Strategy.from_periods(args.periods)
else:
    strategy = Strategy.from_duration(args.duration)

test_frequencies = np.logspace(
    np.log10(args.low_frequency), np.log10(args.high_frequency), args.test_points
)
test_frequencies = np.array(
    [
        Frequency(f).get_actual(xdaq.sampleRate.rate, display_warning=False)
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
                frequency=Frequency(frequency),
                channels=test_channels,
                progress=args.progress > 1,
                strategy=strategy,
                raw_data_return=True
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
    periods=[Frequency(f).get_period(xdaq.sampleRate.rate) for f in test_frequencies],
)
