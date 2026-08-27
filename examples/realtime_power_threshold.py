"""
Closed-loop spectral-power detection on live RHS data, end to end:

    RHS signal
      -> band-pass filter (or Goertzel, for a single frequency)
      -> magnitude squared
      -> integration / averaging over a time window
      -> power-threshold comparison
      -> TTL output

"""
import signal
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from pyxdaq.datablock import Samples, amplifier2uv
from pyxdaq.xdaq import get_XDAQ

# --- settings ---------------------------------------------------------------
TARGET_STREAM = 0  # headstage stream to watch
TARGET_CHANNEL = 0  # channel within that stream
TTL_CHANNEL = 0  # TTL line to drive on detection

BAND = (13.0, 30.0)  # Hz, the band whose power is being tracked
WINDOW_S = 0.1  # integration / averaging window
THRESHOLD = 3e3  # uV^2
HYSTERESIS_RATIO = 0.7  # power must fall to 70% of THRESHOLD before re-arming

TARGET_LATENCY_MS = 1.0  # transport batching ceiling; see the docstring


def _as_2d(x: np.ndarray) -> Tuple[np.ndarray, bool]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return x[:, None], True
    return x, False


@dataclass
class BandpowerDetector:
    """
    Causal IIR band-pass + per-sample leaky integration (EMA). Low latency,
    runs sample-by-sample. Good default for a band of frequencies (e.g. beta,
    13-30 Hz). State carries across calls, so it can be fed callback chunks.
    """
    fs: float
    band: Tuple[float, float] = (13.0, 30.0)
    order: int = 4
    window_s: float = 0.1
    threshold: float = 1.0
    hysteresis_ratio: float = 0.7
    n_channels: int = 1

    def __post_init__(self):
        self._sos = butter(self.order, self.band, btype="bandpass", fs=self.fs, output="sos")
        self._filt_zi = np.zeros((self._sos.shape[0], 2, self.n_channels))

        # power[n] = (1-a)*power[n-1] + a*x^2[n]: 1-pole low-pass, tau ~= window_s.
        self._alpha = 1.0 - np.exp(-1.0 / (self.window_s * self.fs))
        self._smooth_b = np.array([self._alpha])
        self._smooth_a = np.array([1.0, -(1.0 - self._alpha)])
        self._smooth_zi = np.zeros((1, self.n_channels))

        self._power = np.zeros(self.n_channels)
        self._triggered = np.zeros(self.n_channels, dtype=bool)

    def process(self, chunk: np.ndarray):
        """
        Feed newly-arrived samples, shape (n,) or (n, n_channels). Returns
        (power, triggered) at the *end* of this chunk, one per channel
        (scalars if the input was 1-D).
        """
        chunk2d, squeeze = _as_2d(chunk)
        filtered, self._filt_zi = sosfilt(self._sos, chunk2d, axis=0, zi=self._filt_zi)
        sq = filtered * filtered

        power_trace, self._smooth_zi = lfilter(
            self._smooth_b, self._smooth_a, sq, axis=0, zi=self._smooth_zi
        )
        self._power = power_trace[-1]

        hi, lo = self.threshold, self.threshold * self.hysteresis_ratio
        self._triggered = np.where(self._triggered, self._power > lo, self._power > hi)

        power, triggered = self._power.copy(), self._triggered.copy()
        return (power[0], triggered[0]) if squeeze else (power, triggered)


@dataclass
class GoertzelPowerDetector:
    """
    Single-frequency power (cheaper than a full FFT bin), evaluated once per
    block of `block_size` samples.

    Buffers across calls and does the heavy work only at a block boundary, so
    its cost tracks the sample rate rather than the callback rate -- much
    cheaper than BandpowerDetector at high channel counts and small chunks.
    """
    fs: float
    freq: float
    block_size: int = 256
    window_s: float = 0.1
    threshold: float = 1.0
    hysteresis_ratio: float = 0.7
    n_channels: int = 1

    def __post_init__(self):
        k = round(self.block_size * self.freq / self.fs)
        if k == 0:
            raise ValueError(
                f'block_size={self.block_size} is too small to resolve freq={self.freq} Hz '
                f'at fs={self.fs} Hz (need block_size >= ~{self.fs / self.freq:.0f})'
            )
        w = 2 * np.pi * k / self.block_size
        self._coeff = 2 * np.cos(w)
        self._goertzel_a = np.array([1.0, -self._coeff, 1.0])
        self._leftover = np.zeros((0, self.n_channels))

        # One result per block, so the EMA time constant is in blocks.
        block_rate = self.fs / self.block_size
        self._alpha = 1.0 - np.exp(-1.0 / (self.window_s * block_rate))
        self._smooth_b = np.array([self._alpha])
        self._smooth_a = np.array([1.0, -(1.0 - self._alpha)])
        self._smooth_zi = np.zeros((1, self.n_channels))

        self._power = np.zeros(self.n_channels)
        self._triggered = np.zeros(self.n_channels, dtype=bool)

    def process(self, chunk: np.ndarray) -> List[Tuple]:
        """
        Returns one (power, triggered) per integration block completed in this
        call -- often zero or one.
        """
        chunk2d, squeeze = _as_2d(chunk)
        buf = np.concatenate([self._leftover, chunk2d], axis=0)
        n_blocks = len(buf) // self.block_size

        results = []
        for b in range(n_blocks):
            block = buf[b * self.block_size:(b + 1) * self.block_size]
            # s[n] = x[n] + coeff*s[n-1] - s[n-2] is a 2nd-order IIR that
            # must restart from zero state at every block boundary.
            s, _ = lfilter([1.0], self._goertzel_a, block, axis=0, zi=np.zeros((2, block.shape[1])))
            s_last, s_prev = s[-1], s[-2]
            block_power = s_last**2 + s_prev**2 - self._coeff * s_last * s_prev
            block_power /= self.block_size  # normalize like |DFT bin|^2 / N

            smoothed, self._smooth_zi = lfilter(
                self._smooth_b, self._smooth_a, block_power[None, :], axis=0, zi=self._smooth_zi
            )
            self._power = smoothed[0]

            hi, lo = self.threshold, self.threshold * self.hysteresis_ratio
            self._triggered = np.where(self._triggered, self._power > lo, self._power > hi)

            results.append(
                (self._power[0],
                 self._triggered[0]) if squeeze else (self._power.copy(), self._triggered.copy())
            )

        self._leftover = buf[n_blocks * self.block_size:]
        return results


def main():
    xdaq = get_XDAQ(rhs=True)
    print(xdaq.ports)

    xdaq.set_ttl_override(0xFFFFFFFF)  # host software drives the TTL outputs

    # Swap in GoertzelPowerDetector here if only one frequency matters.
    detector = BandpowerDetector(
        fs=xdaq.sample_rate_hz,
        band=BAND,
        window_s=WINDOW_S,
        threshold=THRESHOLD,
        hysteresis_ratio=HYSTERESIS_RATIO,
    )

    chunk_size = int(xdaq.sample_size_in_bytes() * xdaq.sample_rate_hz * TARGET_LATENCY_MS / 1000)

    is_running = True
    ttl_state = False
    n_callbacks = 0

    def handle_sigint(sig, frame):
        nonlocal is_running
        is_running = False

    signal.signal(signal.SIGINT, handle_sigint)

    def on_error(error: str):
        nonlocal is_running
        if not is_running:  # already shutting down; ignore the drain
            return
        print(f"\n[Callback Error] {error}")
        is_running = False

    def on_samples_received(samples: Samples):
        """
        Runs in the receive thread holding the GIL, so anything slow here
        delays the next chunk. Detection is cheap; printing is not.
        """
        nonlocal ttl_state, n_callbacks
        n_callbacks += 1

        amp_uv = amplifier2uv(samples.amp[:, TARGET_STREAM, TARGET_CHANNEL, 1])
        power, triggered = detector.process(amp_uv)

        # Edge-triggered: only touch the hardware when the decision changes.
        if triggered != ttl_state:
            xdaq.set_ttl_out(TTL_CHANNEL, bool(triggered))
            ttl_state = bool(triggered)
            print(f"\nTTL {'ON ' if ttl_state else 'OFF'} | power={power:12.1f} uV^2")

        if n_callbacks % 200 == 0:
            print(
                f"power={power:12.1f} uV^2 | threshold={THRESHOLD:10.1f}"
                f" | TTL={'ON ' if ttl_state else 'OFF'}",
                end="  \r",
                flush=True,
            )

    print(
        f"chunk={chunk_size} B ({TARGET_LATENCY_MS:.2f} ms)  "
        f"band={BAND[0]:.0f}-{BAND[1]:.0f} Hz  window={WINDOW_S * 1000:.0f} ms\n"
        "Running -- Ctrl+C to stop."
    )

    with xdaq.start_receiving_samples(callbacks=[on_samples_received], on_error=on_error,
                                      chunk_size=chunk_size):
        xdaq.start(continuous=True)
        while is_running:
            time.sleep(0.1)
        xdaq.stop(wait=True)
        # The callback may still fire until the context manager exits.

    if ttl_state:
        xdaq.set_ttl_out(TTL_CHANNEL, False)

    print("\nExiting...")


if __name__ == "__main__":
    main()
