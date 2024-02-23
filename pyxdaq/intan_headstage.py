import math
from functools import partial
from pathlib import Path
from typing import Union

import numpy as np

from .constants import SampleRate
from .register import ISA, RegisterController, machinecode


# Returns the value of the RH1 resistor (in ohms) corresponding to a particular upper
# bandwidth value (in Hz).
def rH1FromUpperBandwidth(upperBandwidth):
    log10f = math.log10(upperBandwidth)
    return 0.9730 * pow(10.0, (8.0968 - 1.1892 * log10f + 0.04767 * log10f * log10f))


# Returns the value of the RH2 resistor (in ohms) corresponding to a particular upper
# bandwidth value (in Hz).
def rH2FromUpperBandwidth(upperBandwidth):
    log10f = math.log10(upperBandwidth)
    return 1.0191 * pow(10.0, (8.1009 - 1.0821 * log10f + 0.03383 * log10f * log10f))


# Returns the value of the RL resistor (in ohms) corresponding to a particular lower
# bandwidth value (in Hz).
def rLFromLowerBandwidth(lowerBandwidth):
    log10f = math.log10(lowerBandwidth)

    if lowerBandwidth < 4.0:
        return 1.0061 * pow(
            10.0, (
                4.9391 - 1.2088 * log10f + 0.5698 * log10f * log10f
                + 0.1442 * log10f * log10f * log10f
            )
        )
    else:
        return 1.0061 * pow(10.0, (4.7351 - 0.5916 * log10f + 0.08482 * log10f * log10f))


# Returns the amplifier upper bandwidth (in Hz) corresponding to a particular value
# of the resistor RH1 (in ohms).
def upperBandwidthFromRH1(rH1):
    a, b, c = 0.04767, -1.1892, 8.0968 - math.log10(rH1 / 0.9730)
    return pow(10.0, ((-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)))


# Returns the amplifier upper bandwidth (in Hz) corresponding to a particular value
# of the resistor RH2 (in ohms).
def upperBandwidthFromRH2(rH2):
    a, b, c = 0.03383, -1.0821, 8.1009 - math.log10(rH2 / 1.0191)
    return pow(10.0, ((-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)))


# Returns the amplifier lower bandwidth (in Hz) corresponding to a particular value
# of the resistor RL (in ohms).
def lowerBandwidthFromRL(rL):
    if rL < 5100.0:
        rL = 5100.0

    if rL < 30000.0:
        a, b, c = 0.08482, -0.5916, 4.7351 - math.log10(rL / 1.0061)
    else:
        a, b, c = 0.3303, -1.2100, 4.9873 - math.log10(rL / 1.0061)
    return pow(10.0, ((-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)))


class IntanHeadstage:

    def __init__(
        self,
        sample_rate: SampleRate,
        register_config: Union[str, Path],
        isa_config: Union[str, Path],
    ):
        self.controller = RegisterController.from_json(Path(register_config).read_text())
        self.isa = ISA.from_json(Path(isa_config).read_text())
        self.encode = partial(machinecode, isa=self.isa, registers=self.controller.registers)
        self.sample_rate = sample_rate

    @classmethod
    def _get_dsp_freq_table(cls, sample_rate):
        x = 2.0**np.arange(1, 16)
        return sample_rate * np.log(x / (x - 1.0)) / (np.pi * 2)

    @classmethod
    def _set_dsp_cutoff_freq(cls, new_dsp_cutoff_freq, sample_rate):
        f_cutoff = cls._get_dsp_freq_table(sample_rate)
        log_new_dsp_cutoff_freq = np.log10(new_dsp_cutoff_freq)

        if new_dsp_cutoff_freq > f_cutoff[0]:
            dsp_cutoff_freq = 0
        elif new_dsp_cutoff_freq < f_cutoff[14]:
            dsp_cutoff_freq = 14
        else:
            log_f_cutoff = np.log10(f_cutoff)
            log_diff = np.abs(log_new_dsp_cutoff_freq - log_f_cutoff)
            dsp_cutoff_freq = np.argmin(log_diff) + 1

        return dsp_cutoff_freq, f_cutoff[dsp_cutoff_freq - 1]

    def set_dsp_cutoff_freq(self, new_dsp_cutoff_freq):
        v, _ = self._set_dsp_cutoff_freq(new_dsp_cutoff_freq, self.sample_rate.value[2])
        self.controller.set('dspCutoffFreq', v)

    @classmethod
    def _upper_bw_reg(cls, upperBandwidth):
        RH1Base = 2200.0
        RH1Dac1Unit = 600.0
        RH1Dac2Unit = 29400.0
        RH1Dac1Steps = 63
        RH1Dac2Steps = 31

        upperBandwidth = min(upperBandwidth, 30000.0)

        rH1Target = rH1FromUpperBandwidth(upperBandwidth)

        rH1Dac1 = 0
        rH1Dac2 = 0
        rH1Actual = RH1Base

        for _ in range(RH1Dac2Steps):
            if (rH1Actual < rH1Target - (RH1Dac2Unit - RH1Dac1Unit / 2)):
                rH1Actual += RH1Dac2Unit
                rH1Dac2 += 1

        for _ in range(RH1Dac1Steps):
            if (rH1Actual < rH1Target - (RH1Dac1Unit / 2)):
                rH1Actual += RH1Dac1Unit
                rH1Dac1 += 1

        RH2Base = 8700.0
        RH2Dac1Unit = 763.0
        RH2Dac2Unit = 38400.0
        RH2Dac1Steps = 63
        RH2Dac2Steps = 31

        rH2Target = rH2FromUpperBandwidth(upperBandwidth)

        rH2Dac1 = 0
        rH2Dac2 = 0
        rH2Actual = RH2Base

        for _ in range(RH2Dac2Steps):
            if rH2Actual < rH2Target - (RH2Dac2Unit - RH2Dac1Unit / 2):
                rH2Actual += RH2Dac2Unit
                rH2Dac2 += 1

        for _ in range(RH2Dac1Steps):
            if rH2Actual < rH2Target - (RH2Dac1Unit / 2):
                rH2Actual += RH2Dac1Unit
                rH2Dac1 += 1

        actualUpperBandwidth1 = upperBandwidthFromRH1(rH1Actual)
        actualUpperBandwidth2 = upperBandwidthFromRH2(rH2Actual)

        # Upper bandwidth estimates calculated from actual RH1 value and acutal RH2 value
        # should be very close; we will take their geometric mean to get a single
        # number.
        actualUpperBandwidth = math.sqrt(actualUpperBandwidth1 * actualUpperBandwidth2)

        return actualUpperBandwidth, rH1Dac1, rH1Dac2, rH2Dac1, rH2Dac2

    @classmethod
    def _lower_bw_reg(cls, lowerBandwidth):
        RLBase = 3500.0
        RLDac1Unit = 175.0
        RLDac2Unit = 12700.0
        RLDac3Unit = 3000000.0
        RLDac1Steps = 127
        RLDac2Steps = 63

        # Lower bandwidths higher than 1.5 kHz don't work well with the Rhs2000 amplifiers
        lowerBandwidth = min(lowerBandwidth, 1500.0)

        rLTarget = rLFromLowerBandwidth(lowerBandwidth)

        rLDac1 = 0
        rLDac2 = 0
        rLDac3 = 0
        rLActual = RLBase

        if lowerBandwidth < 0.15:
            rLActual += RLDac3Unit
            rLDac3 += 1

        for _ in range(RLDac2Steps):
            if (rLActual < rLTarget - (RLDac2Unit - RLDac1Unit / 2)):
                rLActual += RLDac2Unit
                rLDac2 += 1

        for _ in range(RLDac1Steps):
            if rLActual < rLTarget - (RLDac1Unit / 2):
                rLActual += RLDac1Unit
                rLDac1 += 1

        actualLowerBandwidth = lowerBandwidthFromRL(rLActual)

        return actualLowerBandwidth, rLDac1, rLDac2, rLDac3

    def get_zcheck_cmds(self, frequency: float, amplitude: float, register: int, maxlength: int):
        if (amplitude < 0.0) or (amplitude > 128.0):
            raise ValueError("Amplitude out of range.")
        if frequency < 0.0:
            raise ValueError("Negative frequency not allowed.")
        elif frequency > self.sample_rate.rate / 4.0:
            raise ValueError(
                f"Frequency too high relative to sampling rate. {frequency:.2f} > Max: {self.sample_rate.rate / 4.0}"
            )

        if frequency == 0.0:
            return [
                self.encode('writeval', addr=register, value=int(math.floor(amplitude)))
            ] * maxlength
        else:
            period = round(self.sample_rate.rate / frequency)
            if period > maxlength:
                raise ValueError("Frequency too low relative to sampling rate.")

            x = np.sin(np.linspace(0, 2 * np.pi, period)) * amplitude + 128
            x = np.clip(np.round(x), 0, 255)
            return [self.encode('writeval', addr=register, value=v) for v in map(int, x)]
