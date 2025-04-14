import logging
import math
from dataclasses import dataclass
from typing import Union

import numpy as np

from pyxdaq.datablock import amplifier2mv

logger = logging.getLogger(__name__)


def amplitudeOfFreqComponent(data, sample_rate, frequency):
    if len(data) % 2 != 0:
        data = data[:-1]  # Truncate the last sample for even length

    n = np.arange(len(data))
    cos_wave = np.cos(2 * np.pi * frequency * n / sample_rate)
    sin_wave = -np.sin(2 * np.pi * frequency * n / sample_rate)

    # Compute the real and imaginary parts
    Re = np.dot(data, cos_wave)
    Im = np.dot(data, sin_wave)
    r = (Re + 1j * Im) * 2
    return r / len(data)


def measureComplexAmplitude(ampdata: np.ndarray, sampleRate: int, frequency: float) -> np.ndarray:
    if len(ampdata.shape) != 2:
        raise ValueError("ampdata must be in the shape (num_tests, signals)")
    # Measure real (iComponent) and imaginary (qComponent) amplitude of frequency component.
    return np.array([amplitudeOfFreqComponent(i, sampleRate, frequency) for i in ampdata])


def factor_out_parallel_capacitance(
    impedance_magnitude, impedance_phase, frequency, parasitic_capacitance
):
    # Convert from polar coordinates to rectangular coordinates.
    measured_r = impedance_magnitude * np.cos(impedance_phase)
    measured_x = impedance_magnitude * np.sin(impedance_phase)

    cap_term = 2 * math.pi * frequency * parasitic_capacitance
    x_term = cap_term * (measured_r * measured_r + measured_x * measured_x)
    denominator = cap_term * x_term + 2 * cap_term * measured_x + 1
    true_r = measured_r / denominator
    true_x = (measured_x + x_term) / denominator

    # Convert from rectangular coordinates back to polar coordinates.
    impedance_magnitude = np.sqrt(true_r * true_r + true_x * true_x)
    impedance_phase = np.rad2deg(np.arctan2(true_x, true_r))

    return impedance_magnitude, impedance_phase


def approximateSaturationVoltage(actualZFreq, highCutoff):
    if actualZFreq < 0.2 * highCutoff:
        return 5000.0
    else:
        return 5000.0 * np.sqrt(1.0 / (1.0 + np.power(3.3333 * actualZFreq / highCutoff, 4.0)))


def calculate_impedance(
    all_signals: np.ndarray,
    sample_rate: float,
    rhs: bool,
    frequency: float,
    return_cap: bool = False,
):
    if len(all_signals.shape) != 3:
        raise ValueError("all_signals must be in the shape (3, tests, signal_length)")
    caps, tests, signal_length = all_signals.shape
    if caps != 3:
        raise ValueError("all_signals must be in the shape (3, tests, signals_length)")

    res = measureComplexAmplitude(
        amplifier2mv(
            all_signals.reshape((-1, signal_length)),
        ), sample_rate, frequency
    ).reshape((caps, tests))

    cap = np.array([0.1e-12, 1e-12, 10e-12])
    dacVoltageAmplitude = 128 * (1.225 / 256)  # this assumes the DAC amplitude was set to 128
    relativeFreq = frequency / sample_rate
    saturate_voltage = approximateSaturationVoltage(frequency, 7500)
    # find the best cap for each channel by looking at largest cap that doesn't saturate
    best_idx = 2 - np.argmax(np.abs(res[::-1, :]) < saturate_voltage, axis=0)
    saturated_cap3 = np.abs(res[1, :]) / np.abs(res[2, :]) > 0.2
    best_idx -= saturated_cap3 & (best_idx == 2)  # if cap 3 is saturated, use cap 2
    best_cap = cap[best_idx]
    best = np.choose(best_idx, res)
    current = 2 * np.pi * frequency * best_cap * dacVoltageAmplitude
    magnitude = np.abs(best) / current * 1e-6 * (18 * relativeFreq * relativeFreq + 1)

    if rhs:
        parasiticCapacitance = 12.0e-12
    else:
        parasiticCapacitance = 15.0e-12
    magnitude, phase = factor_out_parallel_capacitance(
        magnitude, np.angle(best), frequency, parasiticCapacitance
    )

    if rhs:
        magnitude = magnitude * 1.1

    if return_cap:
        return magnitude, phase, best_idx
    else:
        return magnitude, phase


@dataclass
class Frequency:
    """
    Not every frequency is achievable with the current sample rate. This class helps to find the 
    actual frequency that will be used for testing and the actual period of the testing sine wave.
    """
    target: Union[float, np.ndarray]

    def __post_init__(self):
        if np.any(self.target < 1):
            raise ValueError("Frequency must be greater than 1Hz")

    def get_actual(self, sample_rate: float, display_warning: bool = True) -> float:
        actual = sample_rate / np.round(sample_rate / self.target)
        if np.any(abs(actual - self.target) / self.target > 0.05) and display_warning:
            logger.warning(
                f"Actual testing frequency {actual:.1f}Hz is off by more than 5% from target {self.target:.1f}Hz"
            )
        return actual

    def get_period(self, sample_rate: float) -> Union[int, np.ndarray]:
        if isinstance(self.target, float):
            return int(np.round(sample_rate / self.target))
        else:
            return np.round(sample_rate / self.target).astype(int)


@dataclass
class Strategy:
    """
    Helper class to compute the number of periods per measurement based on 
    the desired test frequency and the minimum duration of the measurement.

    This class is not meant to be instantiated directly, use the class methods
    to create an instance.
    Examples:
        Strategy.auto()
        Strategy.from_periods(10)
    """
    _periods: int = None
    _duration: float = None

    _min_periods: int = 5  # less than 5 periods is not enough for accurate measurement
    _min_duration: float = None

    _max_duration: float = 1

    def __post_init__(self):
        if (self._periods or self._duration or self._min_periods or self._min_duration) is None:
            raise ValueError('At least one parameter must be set')
        if self._periods is not None and self._duration is not None:
            raise ValueError('Only one of periods or duration can be set')

    @classmethod
    def auto(cls):
        """
        Default strategy, 5 periods or 0.02 seconds, whichever is greater.
        For a 1000Hz test frequency, 0.02 seconds will be used because 5 periods is only 0.005 sec.
        """
        return cls(_min_periods=5, _min_duration=0.02)

    @classmethod
    def from_periods(cls, periods: int):
        return cls(_periods=periods)

    @classmethod
    def from_duration(cls, duration_in_second: float):
        return cls(_duration=duration_in_second)

    def get_num_periods(self, frequency: float) -> int:
        min_periods_from_min_duration = 0
        if self._min_duration is not None:
            min_periods_from_min_duration = math.ceil(self._min_duration * frequency)
        min_periods_from_both = 0
        if self._min_periods is not None:
            min_periods_from_both = max(self._min_periods, min_periods_from_min_duration)

        if self._periods is not None:
            periods = max(self._periods, min_periods_from_both)
        elif self._duration is not None:
            periods_from_duration = math.ceil(self._duration * frequency)
            if periods_from_duration < min_periods_from_both:
                raise ValueError(
                    f'Number of periods from duration ({periods_from_duration}) '
                    f'is less than the minimum number of periods ({min_periods_from_both})'
                )
            periods = periods_from_duration
        else:
            periods = min_periods_from_both

        if periods / frequency > self._max_duration:
            raise ValueError(
                f'Measurement duration for {periods} periods at {frequency} Hz '
                f'is greater than the maximum duration ({self._max_duration:.2f} seconds)'
            )
        return periods
