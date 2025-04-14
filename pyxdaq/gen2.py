import math
from typing import Union

import numpy as np

from .xdaq import XDAQ


def upload_waveform(xdaq: XDAQ, waveform: np.ndarray, idx: int, divisor: int):
    if waveform.dtype != np.uint16:
        raise TypeError(f'Only 16 bits resolution for waveform, get {waveform.dtype}')
    for i, val in enumerate(waveform.astype(np.uint32)):
        xdaq.dev.set_register(0x2008 + idx * 8, int(val) | (i << 16) | (1 << 31), 0xFFFFFFFF)
    xdaq.dev.set_register(0x200C + idx * 8, divisor | (len(waveform) - 1) << 16, 0xFFFFFFFF)


def config_waveform(xdaq: XDAQ, enable: int):
    xdaq.dev.set_register(0x2004, enable, 15)  # enable
    xdaq.dev.set_register(0x2000, enable, 15)  # reset enabled


def config_channel(xdaq: XDAQ, channel: int, waveform: Union[int, None]):
    if waveform is None:
        xdaq.dev.set_register(0x2028 + 4 * channel, 0 | (0 << 31), 0xFFFFFFFF)
    else:
        xdaq.dev.set_register(0x2028 + 4 * channel, waveform | (1 << 31), 0xFFFFFFFF)


def get_sine(period, amp=32768):
    return np.clip(
        (32768 + amp * np.sin(np.linspace(0, np.pi * 2, period, endpoint=False))).astype(np.int32),
        0,
        65535,
    ).astype(np.uint16)


def approximate_waveform_param(
    target_freq: float,
    tolerance: float = 0,
    clock: int = int(125e6),
    min_div: int = 1024,
    max_div: int = 65535,
    min_period: int = 32,
    max_period: int = 1024,
):
    d = clock / target_freq
    best = float('inf')
    best_param = None
    for period in range(max_period, min_period, -1):
        a = max(int(math.floor(d / period)), min_div)
        b = min(int(math.ceil(d / period)), max_div)
        for div in range(a, b + 1):
            if d / period == div:
                return (period, div)
            resudial = abs(clock / period / div - target_freq)
            if resudial < tolerance:
                return (period, div)
            if resudial < best:
                best = resudial
                best_param = (period, div)
    return best_param
