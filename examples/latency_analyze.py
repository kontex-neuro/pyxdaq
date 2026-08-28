"""
Plot the latency recorded by `pipeline_latency_detector.py`.

Reads the .npz written by --save-npz: every stage plus the configuration that
produced it, so the plot labels itself and nothing has to be remembered or
passed on the command line.

Each transport record is one sample's `now_us - ttlout_us`: the time from the
host stamping its microsecond counter into the TTL register to that value
coming back through FPGA frame assembly, DMA and data parse.

Two things to know before reading the plots:

  * The bulk spread is NOT jitter. Within one delivered chunk the latency falls
    by exactly one sample period per sample, because every sample in the chunk
    is timed against the same arrival instant while each was assembled a sample
    period apart. At the next chunk it jumps back up. The spread is therefore a
    batching artifact whose width is set by the chunk size, and panel (a) shows
    it directly.
  * Mean +- sigma is meaningless on this distribution -- sigma is set by a
    handful of rare spikes, so mean - 2 sigma can land below the hardware floor.
    Percentiles and a survival curve are used instead.
"""
import argparse
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('path', nargs='?', default='latency.npz')
parser.add_argument(
    '--warmup',
    type=int,
    default=None,
    help='samples to drop from the start; defaults to whatever the run itself '
    'dropped, so plot and console summary agree'
)
parser.add_argument(
    '--spike-us',
    type=float,
    default=None,
    help='outlier threshold; default p99 x 1.5, i.e. just above the batching '
    'sawtooth. A fixed value only suits the chunk size it was chosen for'
)
parser.add_argument('--save', default=None, help='write figure here instead of showing it')
args = parser.parse_args()

path = Path(args.path)
if path.suffix != '.npz':
    raise SystemExit(f'{path}: unrecognized file type, must be .npz')

z = np.load(path, allow_pickle=False)
raw = z['transport'].astype(np.int64)  # already signed
# t_recv and ttlout (pipeline_latency_detector.py) are each masked into one
# 32-bit rolling microsecond epoch, matching the TTL register's width.
# Recenter into that wrapped range so a sample taken across a rollover reads
# correctly.
raw = ((raw + 2**31) % 2**32) - 2**31
if (raw < 0).any():
    i = int(np.flatnonzero(raw < 0)[0])
    raise SystemExit(
        f'{path}: sample {i} = {raw[i]} us is still negative after recentering, not a '
        f'valid latency ({int((raw < 0).sum())} of {len(raw)} samples affected).'
    )
meta = {k: z[k].item() for k in z.files if z[k].ndim == 0}
stages = {k: z[k] for k in ('parsing', 'compute', 'ttl_write') if k in z.files}
recorded_chunk = z['chunk_sizes'] if 'chunk_sizes' in z.files else None

sr = 30000.0
warmup = args.warmup if args.warmup is not None else int(meta.get('warmup', 30000))
data = raw[warmup:]
if len(data) == 0:
    raise SystemExit(f'{path}: nothing left after dropping {warmup} warmup samples')
t = np.arange(len(data)) / sr

pct_keys = (50, 95, 99, 99.9, 99.99)
pct = dict(zip(pct_keys, np.percentile(data, pct_keys)))

# A spike is what sits above the batching structure, so the threshold has to
# scale with it: the sawtooth top moves with chunk size, and a fixed value
# lands inside the core distribution at large chunks (flagging ordinary
# batching as spikes) and above the real tail at small ones.
spike_us = args.spike_us if args.spike_us is not None else pct[99] * 1.5
spikes = data >= spike_us

# Chunk structure: latency rises exactly when a new chunk is delivered, so the
# gaps between rises are the per-chunk sample counts.
rises = np.flatnonzero(np.diff(data) > 0) + 1
samples_per_chunk = np.median(np.diff(rises)) if len(rises) > 2 else float('nan')

if meta:
    print(
        f"{path}: {'RHS' if meta.get('rhs') else 'RHD'}  "
        f"{meta.get('num_streams')} streams  {meta.get('channels')} ch  "
        f"{meta.get('detector')}  fs={sr:.0f} Hz"
        f"{'' if meta.get('gc_enabled') is None else '  gc=' + ('on' if meta['gc_enabled'] else 'off')}"
    )
print(f'{path}: n={len(data)} (dropped {warmup} warmup)')
print(
    f'  min={data.min()} p50={pct[50]:.0f} p95={pct[95]:.0f} p99={pct[99]:.0f} '
    f'p99.9={pct[99.9]:.0f} max={data.max()} us'
)
print(
    f'  >{spike_us:.0f} us{"" if args.spike_us is not None else " (p99 x 1.5)"}: '
    f'{spikes.sum()} ({100 * spikes.mean():.3f}%)'
)
print(
    f'  ~{samples_per_chunk:.1f} samples/chunk -> sawtooth span '
    f'~{(samples_per_chunk - 1) * 1e6 / sr:.0f} us'
)

# Cross-check the sawtooth against what the run actually recorded. Compare
# median with median: delivery is often bimodal (the driver alternates between
# a full chunk and a 1-sample remainder), so its mean sits between the two
# modes and would look like a mismatch against any single detected run length.
if recorded_chunk is not None:
    rec_median = float(np.median(recorded_chunk))
    agree = abs(samples_per_chunk - rec_median) <= max(1.0, 0.25 * rec_median)
    print(
        f'  recorded {rec_median:.1f} median, {recorded_chunk.mean():.1f} mean '
        f'samples/callback -- {"consistent" if agree else "MISMATCH, investigate"}'
    )
    modes, counts = np.unique(recorded_chunk, return_counts=True)
    if len(modes) <= 4:
        share = {int(m): f'{100 * c / len(recorded_chunk):.0f}%' for m, c in zip(modes, counts)}
        print(f'  delivery is quantized: {share}')

# Per-callback stages need the same cut as the per-sample transport, expressed
# in callbacks: without it their `max` is a startup cost too.
warmup_cb = 0
if recorded_chunk is not None:
    warmup_cb = int(np.searchsorted(np.cumsum(recorded_chunk), warmup, side='right'))

for name, v in stages.items():
    # ttl_write is benchmarked after acquisition, so no warmup applies to it.
    v = np.asarray(v)[0 if name == 'ttl_write' else warmup_cb:] * 1e6
    if len(v):
        print(
            f'  {name:<10s} p50={np.percentile(v, 50):8.1f}  '
            f'p95={np.percentile(v, 95):8.1f}  max={v.max():9.1f} us'
        )

fig, ax = plt.subplots(2, 2, figsize=(15, 9))

# (a) The batching sawtooth -- invisible if you plot all N points at once.
a = ax[0, 0]
z_ = data[:120]
a.plot(z_, marker='o', ms=3.5, lw=1, color='#2b6cb0')
for i, boundary in enumerate(np.flatnonzero(np.diff(z_) > 0) + 1):
    a.axvline(
        boundary,
        color='#e53e3e',
        lw=.8,
        ls='--',
        alpha=.7,
        label='new chunk delivered' if i == 0 else None
    )
a.set(
    xlabel='sample index',
    ylabel='latency (us)',
    title=f'(a) Batching sawtooth: -{1e6 / sr:.0f} us/sample within a chunk\n'
    f'transfer artifact (~{samples_per_chunk:.1f} samples/chunk), NOT random jitter'
)
a.legend(fontsize=8, loc='upper left')
a.grid(alpha=.3)

# (b) Rolling percentiles -- keeps N points readable and shows spike bursts.
b = ax[0, 1]
nwin = min(200, max(10, len(data) // 200))
edges = np.linspace(0, len(data), nwin + 1).astype(int)
mid = np.array([t[s:e].mean() for s, e in zip(edges[:-1], edges[1:]) if e > s])
roll_keys = (50, 95, 99)
roll_windows = [np.percentile(data[s:e], roll_keys) for s, e in zip(edges[:-1], edges[1:]) if e > s]
roll = dict(zip(roll_keys, np.array(roll_windows).T))
b.fill_between(mid, roll[50], roll[99], color='#bee3f8', label='p50-p99')
for p, col in ((50, '#2b6cb0'), (95, '#dd6b20'), (99, '#c53030')):
    b.plot(mid, roll[p], lw=1.3, color=col, label=f'rolling p{p}')
b.plot(
    t[spikes],
    data[spikes],
    '.',
    color='#e53e3e',
    ms=4,
    alpha=.6,
    label=f'>{spike_us:.0f} us (n={spikes.sum()}, {100 * spikes.mean():.3f}%)'
)
b.set(
    xlabel='time (s)',
    ylabel='latency (us)',
    yscale='log',
    title='(b) Latency over time -- spikes arrive in bursts'
)
b.set_ylim(top=data.max() * 4)  # headroom so the legend clears the spikes
b.legend(fontsize=7, loc='upper right', ncol=2, framealpha=.9)
b.grid(alpha=.3, which='both')

# (c) Core distribution: flat-topped, because phase within a chunk is uniform.
c = ax[1, 0]
# Cut just above the sawtooth top, not at a high percentile: the tail runs to
# ~3000 us and would squash the 85-207 us structure into the left margin.
hi = pct[99] * 1.5
cd = data[data < hi]
if len(cd):
    c.hist(cd, bins=np.arange(cd.min(), cd.max() + 2), color='#90cdf4', edgecolor='none')
else:
    c.text(0.5, 0.5, 'no samples below cutoff', ha='center', va='center', transform=c.transAxes)
for p, col in ((50, '#2b6cb0'), (95, '#dd6b20'), (99, '#c53030')):
    c.axvline(pct[p], color=col, lw=1.6, label=f'p{p} = {pct[p]:.0f} us')
c.set(
    xlabel='latency (us)',
    ylabel='count',
    title=f'(c) Core distribution ({100 * len(cd) / len(data):.1f}% of samples)\n'
    'flat-topped = uniform phase within a chunk'
)
c.legend(fontsize=8, loc='upper right')
c.grid(alpha=.3)

# (d) Survival function -- how to read a latency tail without assuming Gaussian.
d_ = ax[1, 1]
sd = np.sort(data)
d_.plot(sd, 1.0 - np.arange(len(sd)) / len(sd), lw=1.8, color='#2b6cb0')
for p, col in ((50, '#2b6cb0'), (95, '#dd6b20'), (99, '#c53030'), (99.9, '#805ad5')):
    d_.axvline(pct[p], color=col, ls='--', lw=1.2, alpha=.8, label=f'p{p} = {pct[p]:.0f} us')
d_.set(
    xlabel='latency (us)',
    ylabel='P(latency > x)',
    xscale='log',
    yscale='log',
    title='(d) Tail / survival function\n(no Gaussian assumption)'
)
d_.legend(fontsize=8, loc='upper right')
d_.grid(alpha=.3, which='both')

mean, std = data.mean(), data.std()
config = ''
if meta:
    config = (
        f"   |   {'RHS' if meta.get('rhs') else 'RHD'} "
        f"{meta.get('channels')} ch, {meta.get('detector')}, "
        f"chunk {meta.get('hw_chunk')} B"
    )
fig.suptitle(
    f'{platform.system()} latency: Register -> FPGA -> Data Stream -> Data Parse{config}\n'
    f'n={len(data)}  min={data.min()}  p50={pct[50]:.0f}  p95={pct[95]:.0f}  '
    f'p99.9={pct[99.9]:.0f}  max={data.max()} us   '
    f'(mean={mean:.0f}+-{std:.0f} for reference only: heavily skewed, '
    f'mean-2sigma={mean - 2 * std:.0f} us is below the {data.min()} us floor)',
    fontsize=10
)
plt.tight_layout()
if args.save:
    fig.savefig(args.save, dpi=110)
    print(f'  saved {args.save}')
else:
    plt.show()
