import numpy as np

from .constants import SampleRate
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
        if sample_rate < 3334.0:
            return 40, 32
        if sample_rate < 4001.0:
            return 40, 16
        if (sample_rate < 5001.0):
            return 40, 8
        if (sample_rate < 6251.0):
            return 32, 8
        if (sample_rate < 8001.0):
            return 26, 8
        if (sample_rate < 10001.0):
            return 18, 4
        if (sample_rate < 12501.0):
            return 16, 3
        if (sample_rate < 15001.0):
            return 7, 3
        return 4, 2

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
        cmd.append(self.encode('clear'))
        cmd.extend([self.encode('dummy')] * 10)
        return cmd

    @_pack_instructions
    def dummy(self, n):
        return [self.encode('dummy')] * n
