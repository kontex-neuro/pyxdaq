from typing import Union

import numpy as np

from .constants import SampleRate, StimStepSize
from .intan_headstage import IntanHeadstage


def _pack_instructions(func):

    def wrapper(self, *args, **kwargs):
        return np.array(func(self, *args, **kwargs), dtype=self.isa.dtype)

    return wrapper


class RHSDriver(IntanHeadstage):

    def __init__(self, sample_rate: SampleRate, register_config, isa_config):
        super().__init__(
            sample_rate=sample_rate,
            register_config=register_config,
            isa_config=isa_config,
        )

        muxBias, adcBufferBias = self._sample_rate_reg(self.sample_rate)
        self.controller.set('muxBias', muxBias)
        self.controller.set('adcBufferBias', adcBufferBias)

    def set_upper_bandwidth(self, upper_bandwidth):
        actual, rH1Dac1, rH1Dac2, rH2Dac1, rH2Dac2 = self._upper_bw_reg(upper_bandwidth)
        self.controller.set('rh1Sel1', rH1Dac1)
        self.controller.set('rh1Sel2', rH1Dac2)
        self.controller.set('rh2Sel1', rH2Dac1)
        self.controller.set('rh2Sel2', rH2Dac2)
        return actual

    def set_lower_bandwidth_a(self, lower_bandwidth):
        actural, rLDac1, rLDac2, rLDac3 = self._lower_bw_reg(lower_bandwidth)
        self.controller.set('rlASel1', rLDac1)
        self.controller.set('rlASel2', rLDac2)
        self.controller.set('rlASel3', rLDac3)
        return actural

    def set_lower_bandwidth_b(self, lower_bandwidth):
        actural, rLDac1, rLDac2, rLDac3 = self._lower_bw_reg(lower_bandwidth)
        self.controller.set('rlBSel1', rLDac1)
        self.controller.set('rlBSel2', rLDac2)
        self.controller.set('rlBSel3', rLDac3)
        return actural

    @staticmethod
    def _sample_rate_reg(sample_rate: SampleRate):
        sample_rate = sample_rate.value[2]
        if sample_rate < 6001.0:
            return 40, 32
        if sample_rate < 7001.0:
            return 40, 16
        if (sample_rate < 8751.0):
            return 40, 8
        if (sample_rate < 11001.0):
            return 32, 8
        if (sample_rate < 14001.0):
            return 26, 8
        if (sample_rate < 17501.0):
            return 18, 4
        if (sample_rate < 22001.0):
            return 16, 3
        return 5, 3

    @_pack_instructions
    def createCommandListRegisterConfig(self, update_stim: bool, readonly: bool):
        cmd = []
        cmd.extend([self.encode('dummy')] * 2)
        if readonly:
            cmd.extend([self.encode('dummy')] * 54)
        else:
            cmd.extend(self.encode('write', addr=addr) for addr in [0, 1, 2])
            cmd.extend(self.encode('write', addr=addr) for addr in range(4, 9))
            cmd.extend(self.encode('write', addr=addr) for addr in [10, 12, 32, 33])
            if update_stim:
                cmd.extend(self.encode('write', addr=addr) for addr in [34, 35, 36, 37])
            else:
                cmd.extend([self.encode('dummy')] * 4)
            cmd.append(self.encode('write', addr=38))
            cmd.append(self.encode('writem', addr=40))
            cmd.extend(self.encode('write', addr=addr) for addr in [42, 44, 46, 48])
            if update_stim:
                cmd.extend(self.encode('write', addr=addr) for addr in range(64, 80))
                cmd.extend(self.encode('write', addr=addr) for addr in range(96, 111))
                cmd.append(self.encode('writeu', addr=111))
            else:
                cmd.extend([self.encode('dummy')] * 32)

        cmd.extend(self.encode('read', addr=addr) for addr in [255, 254, 253, 252, 251])
        cmd.extend(self.encode('read', addr=addr) for addr in range(0, 9))
        cmd.extend(self.encode('read', addr=addr) for addr in [10, 12])
        cmd.extend(self.encode('read', addr=addr) for addr in range(32, 39))
        cmd.extend(self.encode('read', addr=addr) for addr in [40, 42, 44, 46, 48, 50])
        cmd.extend(self.encode('read', addr=addr) for addr in range(64, 80))
        cmd.extend(self.encode('read', addr=addr) for addr in range(96, 112))
        if readonly:
            cmd.append(self.encode('dummy'))
        else:
            cmd.append(self.encode('clear'))
        cmd.extend([self.encode('dummy')] * 10)
        return cmd

    @_pack_instructions
    def dummy(self, n):
        return [self.encode('dummy')] * n

    @_pack_instructions
    def createCommandListZcheckDac(self, frequency: float, amplitude: float, maxlength: int):
        return self.get_zcheck_cmds(frequency, amplitude, 3, maxlength)

    @_pack_instructions
    def createCommandListSetStimMagnitudes(
        self, magnitude_neg: Union[int, np.ndarray], magnitude_pos: Union[int, np.ndarray],
        channel: int
    ):
        cmd = [self.encode('dummy')] * 2
        if channel is None:
            pass
        else:
            self.controller.set(f'negativeCurrentTrim{channel}', 0x80)
            self.controller.set(f'positiveCurrentTrim{channel}', 0x80)
            self.controller.set(f'negativeCurrentMagnitude{channel}', magnitude_neg)
            self.controller.set(f'positiveCurrentMagnitude{channel}', magnitude_pos)
            cmd.append(self.encode('writeu', addr=96 + channel))
            cmd.append(self.encode('writeu', addr=64 + channel))

        cmd.extend([self.encode('dummy')] * (128 - len(cmd)))
        return cmd

    def setStimStepSize(self, step_size: StimStepSize):
        s1, s2, s3, p, n = {
            StimStepSize.StimStepSizeMin: (127, 63, 3, 6, 6),
            StimStepSize.StimStepSize10nA: (64, 19, 3, 6, 6),
            StimStepSize.StimStepSize20nA: (40, 40, 1, 7, 7),
            StimStepSize.StimStepSize50nA: (64, 40, 0, 7, 7),
            StimStepSize.StimStepSize100nA: (30, 20, 0, 7, 7),
            StimStepSize.StimStepSize200nA: (25, 10, 0, 8, 8),
            StimStepSize.StimStepSize500nA: (101, 3, 0, 9, 9),
            StimStepSize.StimStepSize1uA: (98, 1, 0, 10, 10),
            StimStepSize.StimStepSize2uA: (94, 0, 0, 11, 11),
            StimStepSize.StimStepSize5uA: (38, 0, 0, 14, 14),
            StimStepSize.StimStepSize10uA: (15, 0, 0, 15, 15),
            StimStepSize.StimStepSizeMax: (0, 0, 0, 15, 15)
        }[step_size]
        self.controller.set('stepSel1', s1)
        self.controller.set('stepSel2', s2)
        self.controller.set('stepSel3', s3)
        self.controller.set('stimPbias', p)
        self.controller.set('stimNbias', n)
