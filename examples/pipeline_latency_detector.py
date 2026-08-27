"""
Measures the closed-loop pipeline of `realtime_power_threshold.py` on real
hardware, stage by stage, and reports the end-to-end detection latency. The
detectors are imported from that file, so this measures exactly the code the
demo runs:

    RHS signal -> host -> parsing -> detector -> TTL output

It reports two quantities separately, because different things limit them:

  software round-trip   receive a sample, parse it, run the detector, write a
                        TTL output. Set by transport, chunking and CPU speed.
                        Microseconds.

  detector response     how long after band activity begins the power estimate
                        crosses threshold. Set by the band-pass group delay and
                        the integration window -- the detector's tuning, fixed
                        at realtime_power_threshold.py's defaults, not the CPU.
                        Milliseconds.

HOW "sampling -> host" IS MEASURED
----------------------------------
A single host clock cannot time it: the capture instant exists only on the
FPGA's clock, and a Device-Timestamp fit recovers drift but never the absolute
offset. So, as in libxdaq's `check_latency`, a writer thread stamps the host's
elapsed-microsecond counter into the 32-bit TTL register, and every sample
carries it back in `ttlout`; latency is `now_us - ttlout`, both read on the
host clock.

The writer therefore owns the TTL register throughout, so this script cannot
also drive TTL on detection -- the detector still runs, but `ttl write` is
timed separately as a micro-benchmark of the same register write. It drives the
physical TTL lines with a fast-changing pattern; disconnect anything sensitive.

TWO THINGS THAT WILL SILENTLY RUIN THIS IN PYTHON
-------------------------------------------------
1. The writer thread MUST yield. `set_register_sync` is an mmap'd BAR store, so
   an unthrottled loop starves the GIL: the callback cannot run, the queue
   overflows, and the stream returns 0xFF underflow garbage that parses as
   plausible but wrong latencies. Throttling costs nothing -- the write rate
   does not affect the measured median.
2. This script records an array and three numbers per callback, so its own
   tracked-object count grows all run and gen-2 pauses grow with it. Any spikes
   that caused would be the recording, not the pipeline, so GC is off across the
   measurement window; `--gc` leaves it on to compare.
"""
import argparse
import gc
import sys
import threading
import time

import numpy as np

from pyxdaq.datablock import DataBlock, amplifier2uv
from pyxdaq.xdaq import get_XDAQ

from realtime_power_threshold import BandpowerDetector, GoertzelPowerDetector

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-p', '--period', type=float, default=10.0, help='measurement seconds')
parser.add_argument(
    '-c',
    '--channels',
    type=int,
    default=32,
    help='amplifier channels, and streams enabled to carry them: 32 per RHD '
    'stream, 16 per RHS stream (AC only) -- must be a multiple of that, up '
    'to 1024 on RHD or 128 on RHS'
)
parser.add_argument('-d', '--detector', choices=('bandpass', 'goertzel'), default='bandpass')
parser.add_argument('--rhs', action='store_true', help='use RHS mode')
parser.add_argument(
    '--target-latency-ms',
    type=float,
    default=1.0,
    help='transport chunk size, as the batching latency it implies. Lower '
    'trades transport latency for callback rate; floor is one sample. '
    'Matches realtime_power_threshold.py by default'
)
parser.add_argument(
    '--block-size',
    type=int,
    default=1500,
    help='goertzel integration block; buffered across callbacks, so it is '
    'independent of chunk size and sets how often a result is emitted. '
    'Must resolve freq=20 Hz at fs=30000 Hz (block_size >= ~1500); the '
    'default is an exact bin (30000/1500 = 20 Hz)'
)
parser.add_argument(
    '--skip-response',
    action='store_true',
    help='skip the offline response measurement. It costs ~4 s per run for '
    'bandpass and depends only on the detector configuration, not on chunk '
    'size or channel count -- so measure it once, then skip it when repeating '
    'a run for tail statistics'
)
parser.add_argument(
    '--warmup',
    type=float,
    default=1.0,
    help='seconds to drop before summarizing. Startup costs milliseconds -- '
    'first-touch page faults, cold caches, the driver settling -- and those '
    'samples otherwise set `max` and hide the steady-state tail. --save-npz '
    'keeps the full run either way'
)
parser.add_argument(
    '--gc',
    action='store_true',
    help='leave the cyclic collector running during the measurement window. '
    'Off by default; use this to A/B how much of the tail is GC'
)
parser.add_argument(
    '--save-npz',
    nargs='?',
    const='latency.npz',
    default=None,
    help='every stage plus the configuration, for latency_analyze.py'
)
args = parser.parse_args()

per_stream = 16 if args.rhs else 32
if args.channels < 1 or args.channels % per_stream != 0:
    parser.error(
        f'--channels {args.channels} must be a positive multiple of {per_stream} '
        f'({"RHS" if args.rhs else "RHD"} channels per stream)'
    )
num_streams = args.channels // per_stream

# Faster GIL handoff between the writer thread and the data callback.
sys.setswitchinterval(0.0005)

xdaq = get_XDAQ(rhs=args.rhs, skip_headstage=True)
available_streams = len(xdaq.ports.streams)
if num_streams > available_streams:
    parser.error(
        f'--channels {args.channels} needs {num_streams} streams, but only '
        f'{available_streams} are available on {"RHS" if xdaq.rhs else "RHD"}'
    )
for s in range(available_streams):
    xdaq.config_data_stream(s, s < num_streams)

xdaq.set_ttl_override(0xFFFFFFFF)

sample_size = xdaq.sample_size_in_bytes()
num_streams = xdaq.num_enabled_datastream
fs = xdaq.sample_rate_hz

hw_chunk = max(sample_size, int(sample_size * fs * args.target_latency_ms / 1000))

print(
    f'{"RHS" if xdaq.rhs else "RHD"}  sample={sample_size} B  '
    f'streams={num_streams}/{available_streams}  fs={fs:.0f} Hz  '
    f'{sample_size * fs / 1e6:.1f} MB/s  '
    f'detector={args.detector} on {args.channels} ch'
)
print(
    f'chunk={hw_chunk} B = {hw_chunk / sample_size:.1f} samples '
    f'({hw_chunk * 1e3 / (sample_size * fs):.3f} ms of batching)'
)


# Fixed at realtime_power_threshold.py's own defaults: these change the
# offline response measurement (ms) but not the live compute cost (us) --
# sosfilt/lfilter cost the same regardless of the coefficient values.
_BAND = (13.0, 30.0)
_FREQ = 20.0
_WINDOW_S = 0.1


def make_detector(n_channels):
    if args.detector == 'bandpass':
        return BandpowerDetector(
            fs=fs, band=_BAND, window_s=_WINDOW_S, threshold=np.inf, n_channels=n_channels
        )
    return GoertzelPowerDetector(
        fs=fs,
        freq=_FREQ,
        block_size=args.block_size,
        window_s=_WINDOW_S,
        threshold=np.inf,
        n_channels=n_channels
    )


# --- live measurement -------------------------------------------------------
start = time.perf_counter()
running = True
transport, parsing, compute, chunk_sizes = [], [], [], []
stats = {'callbacks': 0, 'writes': 0, 'errors': 0, 'bad_magic': 0}
ttl_addr = xdaq.ep.WireInTtlOut.value

live_detector = make_detector(args.channels)


def writer():
    """Stamp elapsed microseconds into the TTL register as fast as is safe."""
    set_register = xdaq.dev.raw.set_register_sync
    perf_counter = time.perf_counter
    sleep = time.sleep
    n = 0
    while running:
        set_register(ttl_addr, int((perf_counter() - start) * 1e6) & 0xFFFFFFFF, 0xFFFFFFFF)
        n += 1
        sleep(0)  # release the GIL so the data callback can run
    stats['writes'] = n


def on_data(data, error):
    if error is not None:
        stats['errors'] += 1
        return
    if data is None:
        return

    t_recv = int((time.perf_counter() - start) * 1e6) & 0xFFFFFFFF
    stats['callbacks'] += 1

    buffer = data.numpy
    n = len(buffer) // sample_size
    if n == 0:
        return

    t0 = time.perf_counter()
    try:
        samples = DataBlock.from_buffer(
            xdaq.rhs, sample_size, buffer, num_streams, xdaq.device_timestamp
        ).to_samples()
    except ValueError:  # bad magic -- underflow garbage from a starved writer
        stats['bad_magic'] += 1
        return
    t1 = time.perf_counter()
    parsing.append(t1 - t0)

    transport.append(t_recv - samples.ttlout.ravel().astype(np.int64))

    # RHD amp is (n, streams, 32); RHS is (n, streams, 16, 2) = (DC, AC).
    amp = samples.amp[..., 1] if xdaq.rhs else samples.amp
    live_detector.process(amplifier2uv(amp.reshape(n, -1)))
    compute.append(time.perf_counter() - t1)
    chunk_sizes.append(n)


thread = threading.Thread(target=writer, daemon=True)

# Collect once so the window starts clean, then freeze everything alive now
# into the permanent generation -- without freeze, re-enabling later would
# rescan the whole interpreter in one pause.
if not args.gc:
    gc.collect()
    gc.freeze()
    gc.disable()
gc_before = [g['collections'] for g in gc.get_stats()]  # after the setup collect

stream = xdaq.dev.start_receiving_aligned_buffer(
    xdaq.ep.PipeOutData, sample_size, on_data, chunk_size=hw_chunk
)
with stream:
    xdaq.start(continuous=True)
    thread.start()
    time.sleep(args.period)
    running = False
    xdaq.stop(wait=True)
thread.join(timeout=1)

# The register write a real trigger would issue, timed separately because the
# writer thread owned the register during acquisition.
ttl_write = []
for _ in range(2000):
    t = time.perf_counter()
    xdaq.set_ttl_out(0, False)
    ttl_write.append(time.perf_counter() - t)

# Window closed; the offline section below is untimed, so let GC run again.
gc_collections = [a - b for a, b in zip([g['collections'] for g in gc.get_stats()], gc_before)]
if not args.gc:
    gc.unfreeze()
    gc.enable()


# --- offline: how long until the detector actually reacts -------------------
def feed(detector, x, step):
    """Push x through `detector` in `step`-sample chunks -> (time_s, power)."""
    times, powers = [], []
    for i in range(0, len(x) - step + 1, step):
        out = detector.process(x[i:i + step])
        # Bandpower returns one (power, triggered); Goertzel a list of them.
        for power, _ in (out if isinstance(out, list) else [out]):
            powers.append(float(np.ravel(power)[0]))
            times.append((i + step) / fs)  # attribute to the end of the chunk
    return np.asarray(times), np.asarray(powers)


_TONE_UV = 50.0  # test tone amplitude vs 20 uV RMS noise
_TRIALS = 20  # repetitions, for a p50/p90 rather than a single draw


def measure_response():
    """
    Time from the onset of band activity to the power estimate crossing
    threshold, for the configured detector.

    The threshold is calibrated per trial from the pre-onset baseline (p99.9
    x 1.5), so the result reflects the configured band and window rather than a
    threshold mis-scaled for the test tone. A stricter threshold or a weaker
    signal gives a longer time.

    Fed in fine steps rather than at the transport chunk size, so this is the
    purely algorithmic delay -- waiting for a chunk to fill is already counted
    in sampling->host.
    """
    step = max(1, round(fs / 5000))  # ~0.2 ms resolution
    pre = int(1.0 * fs)
    t_tone = np.arange(int(1.0 * fs)) / fs
    tone = _TONE_UV * np.sin(2 * np.pi * _FREQ * t_tone)

    latencies = []
    for seed in range(_TRIALS):
        rng = np.random.default_rng(1000 + seed)
        x = rng.normal(0, 20, pre + len(tone))
        x[pre:] += tone

        times, powers = feed(make_detector(1), x, step)
        onset = pre / fs
        baseline = powers[times < onset]
        if len(baseline) < 10:
            continue
        crossings = np.flatnonzero(
            (times >= onset) & (powers > np.percentile(baseline, 99.9) * 1.5)
        )
        if len(crossings):
            latencies.append(times[crossings[0]] - onset)
    return np.asarray(latencies)


response_s = np.asarray([]) if args.skip_response else measure_response()

# --- report -----------------------------------------------------------------
transport_us = np.concatenate(transport)

# The saved arrays stay whole -- latency_analyze.py drops its own warmup, and
# trimming here as well would drop it twice. Only the summary below is trimmed,
# so `warmup` travels in the npz for the two to agree by construction.
warmup_samples = min(int(args.warmup * fs), len(transport_us))
# Same cut expressed in callbacks: how many were needed to cover those samples.
warmup_callbacks = int(np.searchsorted(np.cumsum(chunk_sizes), warmup_samples, side='right'))
live_s = max(args.period - args.warmup, 1e-9)

if warmup_callbacks >= len(compute):
    raise SystemExit(
        f'--warmup {args.warmup} s leaves nothing of a {args.period} s run to summarize'
    )

if args.save_npz:
    # Signed, so no wrap to undo, and self-describing so latency_analyze.py
    # needs nothing passed on the command line.
    np.savez_compressed(
        args.save_npz,
        transport=transport_us,
        parsing=np.asarray(parsing),
        compute=np.asarray(compute),
        ttl_write=np.asarray(ttl_write),
        chunk_sizes=np.asarray(chunk_sizes),
        response=response_s,
        fs=fs,
        rhs=xdaq.rhs,
        sample_size=sample_size,
        num_streams=num_streams,
        channels=args.channels,
        detector=args.detector,
        hw_chunk=hw_chunk,
        band_low=_BAND[0],
        band_high=_BAND[1],
        freq=_FREQ,
        window=_WINDOW_S,
        block_size=args.block_size,
        period=args.period,
        gc_enabled=args.gc,
        warmup=warmup_samples,
        # Run health, so pooling many runs can drop the bad ones without
        # re-reading the logs.
        errors=stats['errors'],
        bad_magic=stats['bad_magic'],
        callbacks=stats['callbacks'],
    )
    print(f'\nwrote {args.save_npz} -- plot with:')
    print(f'  python examples/latency_analyze.py {args.save_npz}')

expected = int(args.period * fs)
received = len(transport_us)
print(
    f"\ncallbacks={stats['callbacks']}  samples={received}/{expected} "
    f"({100 * received / expected:.1f}%)  errors={stats['errors']}  "
    f"bad_magic={stats['bad_magic']}  writer={stats['writes'] / args.period / 1000:.1f}k/s"
)
# chunk_size is an upper bound -- the driver hands over whatever is ready,
# usually about half, so real batching is lower than --target-latency-ms.
if chunk_sizes:
    delivered = float(np.mean(chunk_sizes))
    print(
        f'delivered {delivered:.1f} samples/callback of the {hw_chunk / sample_size:.1f} '
        f'requested ({delivered * 1e3 / fs:.3f} ms of actual batching)'
    )
gen_counts = '/'.join(str(c) for c in gc_collections)
print(
    f'gc {"enabled" if args.gc else "disabled"} during the window, '
    f'{gen_counts} collections by generation'
)
if stats['errors'] or stats['bad_magic']:
    print('WARNING: dropped or corrupt data -- results below are unreliable.')
    print('         Raise --target-latency-ms, or lower --channels.')


def summarize(name, values_us):
    a = np.asarray(values_us)
    print(
        f'  {name:<20s} n={len(a):8d}  p50={np.percentile(a, 50):9.1f}  '
        f'p95={np.percentile(a, 95):9.1f}  p99.9={np.percentile(a, 99.9):9.1f}  '
        f'max={a.max():9.1f}'
    )


# Steady state only. Startup lands entirely in the first samples and would
# otherwise own `max`, reading as a spike the pipeline never actually has.
transport_live = transport_us[warmup_samples:]
parsing_live = np.asarray(parsing[warmup_callbacks:])
compute_live = np.asarray(compute[warmup_callbacks:])
ttl_write_live = np.asarray(ttl_write)  # measured after acquisition; no warmup

print(
    f'\n--- software, microseconds ({live_s:.0f} s steady state, '
    f'first {args.warmup:.1f} s dropped) ---'
)
summarize('sampling->host', transport_live)
summarize('parsing', parsing_live * 1e6)
summarize('compute (detector)', compute_live * 1e6)
summarize('ttl write', ttl_write_live * 1e6)

software_us = np.percentile(transport_live, 50) + 1e6 * (
    np.percentile(parsing_live, 50) + np.percentile(compute_live, 50)
    + np.percentile(ttl_write_live, 50)
)
busy = float(np.sum(parsing_live) + np.sum(compute_live)) / live_s
print(f'  {"round-trip (p50 sum)":<20s} {software_us:9.1f} us')
print(f'  {"CPU used":<20s} {100 * busy:9.1f} % of one core')

if args.skip_response:
    print('\n--- detector response skipped (--skip-response) ---')
    response_ms = float('nan')
else:
    print('\n--- detector response, offline, same configuration ---')
    if args.detector == 'bandpass':
        config = f'band={_BAND[0]:.0f}-{_BAND[1]:.0f} Hz'
    else:
        config = f'freq={_FREQ:.0f} Hz  block={args.block_size}'
    print(
        f'  {config}  window={_WINDOW_S * 1000:.0f} ms  '
        f'tone={_FREQ:.0f} Hz @ {_TONE_UV:.0f} uV vs 20 uV noise'
    )
if not args.skip_response and len(response_s) < _TRIALS // 2:
    print(
        f'  detected in only {len(response_s)}/{_TRIALS} trials -- '
        'tone too weak for this band, or window too long'
    )
    response_ms = float('nan')
elif not args.skip_response:
    response_ms = float(np.percentile(response_s, 50) * 1000)
    print(
        f'  {"onset -> crossing":<20s} p50={response_ms:9.2f} ms  '
        f'p90={np.percentile(response_s, 90) * 1000:9.2f} ms  (n={len(response_s)})'
    )

if not np.isnan(response_ms):
    total_ms = response_ms + software_us / 1000
    print('\n--- end to end ---')
    print(
        f'  {total_ms:.2f} ms  =  {response_ms:.2f} ms detector  +  '
        f'{software_us / 1000:.2f} ms software '
        f'({100 * software_us / 1000 / total_ms:.1f}% software)'
    )
