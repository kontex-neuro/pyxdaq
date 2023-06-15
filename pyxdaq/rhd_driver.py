from .constants import SampleRate
from .intan_headstage import IntanHeadstage


class RHDDriver(IntanHeadstage):

    def __init__(self, sample_rate: SampleRate, register_config, isa_config):
        super().__init__(
            sample_rate=sample_rate,
            register_config=register_config,
            isa_config=isa_config,
        )

        muxBias, adcBufferBias = self._sample_rate_reg(self.sample_rate)
        self.controller.set('muxBias', muxBias)
        self.controller.set('adcBufferBias', adcBufferBias)

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

    def set_upper_bandwidth(self, upper_bandwidth):
        actual, rH1Dac1, rH1Dac2, rH2Dac1, rH2Dac2 = self._upper_bw_reg(upper_bandwidth)
        self.controller.set('rH1Dac1', rH1Dac1)
        self.controller.set('rH1Dac2', rH1Dac2)
        self.controller.set('rH2Dac1', rH2Dac1)
        self.controller.set('rH2Dac2', rH2Dac2)
        return actual

    def set_lower_bandwidth(self, lower_bandwidth):
        actural, rLDac1, rLDac2, rLDac3 = self._lower_bw_reg(lower_bandwidth)
        self.controller.set('rLDac1', rLDac1)
        self.controller.set('rLDac2', rLDac2)
        self.controller.set('rLDac3', rLDac3)
        return actural

    def createCommandListUpdateDigOut(self):
        cmd = []
        self.controller.set('tempEnable', 1)
        cmd.extend([self.encode('write', addr=3)] * 3)

        self.controller.set('tempS1', 1)
        self.controller.set('tempS2', 0)
        cmd.extend([self.encode('write', addr=3)] * 4)

        self.controller.set('tempS1', 1)
        self.controller.set('tempS2', 1)
        cmd.extend([self.encode('write', addr=3)] * 8)

        self.controller.set('tempS1', 0)
        self.controller.set('tempS2', 1)
        cmd.extend([self.encode('write', addr=3)] * 8)

        self.controller.set('tempS1', 0)
        self.controller.set('tempS2', 0)
        i = self.encode('write', addr=3)
        cmd.extend([i] * 105)
        return cmd

    def createCommandListTempSensor(self):
        cmd = []
        # auxiliary channels (accelerometer)
        # dt = 1 / 30000 = 33.3333 us
        c32_33_34 = [self.encode('convert', channel=c) for c in [32, 33, 34]]

        self.controller.set('tempEnable', 1)
        cmd.extend(c32_33_34)

        self.controller.set('tempS1', 1)
        self.controller.set('tempS2', 0)
        cmd.append(self.encode('write', addr=3))
        cmd.extend(c32_33_34)

        self.controller.set('tempS1', 1)
        self.controller.set('tempS2', 1)
        cmd.append(self.encode('write', addr=3))
        cmd.extend(c32_33_34 + [self.encode('convert', channel=49)] + c32_33_34)

        self.controller.set('tempS1', 0)
        self.controller.set('tempS2', 1)
        cmd.append(self.encode('write', addr=3))
        cmd.extend(c32_33_34 + [self.encode('convert', channel=49)] + c32_33_34)

        self.controller.set('tempS1', 0)
        self.controller.set('tempS2', 0)
        cmd.append(self.encode('write', addr=3))
        cmd.extend(c32_33_34 + [self.encode('convert', channel=48)])

        for _ in range(25):
            cmd.extend(c32_33_34)
            cmd.append(self.encode('dummy'))

        return cmd

    def createCommandListRegisterConfig(self, calibrate: bool):
        cmd = []
        cmd.extend([self.encode('dummy')] * 2)
        cmd.extend(self.encode('write', addr=addr) for addr in [0, 1, 2, 4, 5])
        cmd.extend(self.encode('write', addr=addr) for addr in range(7, 18))
        cmd.extend(self.encode('read', addr=addr) for addr in [63, 62, 61, 60, 59])
        cmd.extend(self.encode('read', addr=addr) for addr in range(48, 56))
        cmd.extend(self.encode('read', addr=addr) for addr in range(40, 45))
        cmd.extend(self.encode('read', addr=addr) for addr in range(0, 18))
        cmd.append(self.encode('calibrate') if calibrate else self.encode('dummy'))
        cmd.extend(self.encode('write', addr=addr) for addr in range(18, 22))
        cmd.extend([self.encode('dummy')] * (128 - len(cmd)))
        return cmd

    def createCommandListZcheckDac(self, frequency: float, amplitude: float, maxlength: int):
        return self.get_zcheck_cmds(frequency, amplitude, 6, maxlength)
