import numpy as np
import math
from pyxdaq.datablock import amplifier2mv

DEGREES_TO_RADIANS = math.pi / 180.0
RADIANS_TO_DEGREES = 180.0 / math.pi
TWO_PI = 2 * math.pi


def amplitudeOfFreqComponent(data, startIndex, endIndex, sampleRate, frequency):
    data_segment = np.array(data[startIndex:endIndex + 1])
    fft_result = np.fft.rfft(data_segment, norm='forward')
    freqs = np.fft.fftfreq(len(data_segment), d=1 / sampleRate)

    # Find the closest frequency bin to the frequency of interest
    target_index = np.argmin(np.abs(freqs - frequency))

    return fft_result[target_index] * 2


def measureComplexAmplitude(
    ampdata: np.ndarray, sampleRate: int, frequency: float, numPeriods: int
) -> np.ndarray:
    if len(ampdata.shape) != 2:
        raise ValueError("ampdata must be in the shape (num_tests, signals)")
    period = int(sampleRate / frequency)
    startIndex = 0
    endIndex = startIndex + numPeriods * period - 1

    # Move the measurement window to the end of the waveform to ignore start-up transient.
    while endIndex < len(ampdata) - period:
        startIndex += period
        endIndex += period

    # Measure real (iComponent) and imaginary (qComponent) amplitude of frequency component.
    return np.array(
        [amplitudeOfFreqComponent(i, startIndex, endIndex, sampleRate, frequency) for i in ampdata]
    )


def factor_out_parallel_capacitance(
    impedance_magnitude, impedance_phase, frequency, parasitic_capacitance
):
    # Convert from polar coordinates to rectangular coordinates.
    measured_r = impedance_magnitude * np.cos(impedance_phase)
    measured_x = impedance_magnitude * np.sin(impedance_phase)

    cap_term = TWO_PI * frequency * parasitic_capacitance
    x_term = cap_term * (measured_r * measured_r + measured_x * measured_x)
    denominator = cap_term * x_term + 2 * cap_term * measured_x + 1
    true_r = measured_r / denominator
    true_x = (measured_x + x_term) / denominator

    # Convert from rectangular coordinates back to polar coordinates.
    impedance_magnitude = np.sqrt(true_r * true_r + true_x * true_x)
    impedance_phase = RADIANS_TO_DEGREES * np.arctan2(true_x, true_r)

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
    desired_test_frequency: float = 1000,
):
    if len(all_signals.shape) != 3:
        raise ValueError("all_signals must be in the shape (3, tests, signal_length)")
    caps, tests, signal_length = all_signals.shape
    if caps != 3:
        raise ValueError("all_signals must be in the shape (3, tests, signals_length)")

    test_frequency = float(sample_rate / np.round(sample_rate / desired_test_frequency))

    num_periods = int(0.02 * test_frequency)  # 20 ms
    period = sample_rate / test_frequency
    res = measureComplexAmplitude(
        amplifier2mv(
            all_signals.reshape((-1, signal_length)),
        ), sample_rate, test_frequency, num_periods
    ).reshape((caps, tests))

    cap = np.array([0.1e-12, 1e-12, 10e-12])
    dacVoltageAmplitude = 128 * (1.225 / 256)  # this assumes the DAC amplitude was set to 128
    if rhs:
        parasiticCapacitance = 12.0e-12
    else:
        parasiticCapacitance = 15.0e-12
    relativeFreq = test_frequency / sample_rate
    saturate_voltage = approximateSaturationVoltage(test_frequency, 7500)
    # find the best cap for each channel by looking at largest cap that doesn't saturate
    best = 2 - np.argmax(np.abs(res[::-1, :]) < saturate_voltage, axis=0)
    saturated_cap3 = np.abs(res[1, :]) / np.abs(res[2, :]) > 0.2
    best -= saturated_cap3 & (best == 2)
    best_cap = cap[best]
    best = np.choose(best, res)
    current = np.pi * 2 * test_frequency * dacVoltageAmplitude * best_cap
    magnitude = np.abs(best) / current * 1e-6 * (18 * relativeFreq * relativeFreq + 1)
    phase = np.angle(best) + 3 / period
    magnitude, phase = factor_out_parallel_capacitance(
        magnitude, phase, test_frequency, parasiticCapacitance
    )
    if rhs:
        magnitude = magnitude * 1.1
    return magnitude, phase
