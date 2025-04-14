from enum import Enum


class EndPoints(Enum):
    pass


class XDAQWireOut(EndPoints):
    Serial = 0x32
    Hdmi = 0x31
    Fpga = 0x3f
    Daio = 0x30
    Oled = 0x33
    Vido = 0x34
    Expr = 0x35
    Xprt = 0x36


class RHD(EndPoints):
    TTL_override = 0x1400
    WireInResetRun = 0x00
    WireInMaxTimeStep = 0x01
    WireInSerialDigitalInCntl = 0x02
    WireInDataFreqPll = 0x03
    WireInMisoDelay = 0x04
    WireInCmdRamAddr = 0x05
    WireInCmdRamBank = 0x06
    WireInCmdRamData = 0x07
    WireInAuxCmdBank1 = 0x08
    WireInAuxCmdBank2 = 0x09
    WireInAuxCmdBank3 = 0x0a
    WireInAuxCmdLength = 0x0b
    WireInAuxCmdLoop = 0x0c
    WireInLedDisplay = 0x0d
    WireInDacReref = 0x0e
    WireInDataStreamEn = 0x14
    WireInTtlOut = 0x15
    WireInTtlOut32 = 0x10
    WireInDacSource1 = 0x16
    WireInDacSource2 = 0x17
    WireInDacSource3 = 0x18
    WireInDacSource4 = 0x19
    WireInDacSource5 = 0x1a
    WireInDacSource6 = 0x1b
    WireInDacSource7 = 0x1c
    WireInDacSource8 = 0x1d
    WireInDacSource9 = 0x48
    WireInDacSource10 = 0x49
    WireInDacSource11 = 0x4A
    WireInDacSource12 = 0x4B
    WireInDacManual = 0x1e
    WireInMultiUse = 0x1f
    TrigInConfig = 0x40
    TrigInSpiStart = 0x41
    TrigInDacConfig = 0x42
    WireOutNumWords = 0x20
    WireOutSerialDigitalIn = 0x21
    WireOutSpiRunning = 0x22
    WireOutTtlIn = 0x23
    WireOutDataClkLocked = 0x24
    WireOutBoardMode = 0x25
    WireOutBoardId = 0x3e
    WireOutBoardVersion = 0x3f
    PipeOutData = 0xa0
    TrigVStim = 0x40
    TrigMCU = 0x48
    WireOutXDAQStatus = 0x22
    WireInMCUControl = 0x02
    PipeInFirmware = 0x88
    PipeOutFirmware = 0xb0
    WireInSetMode = 0x00
    Enable32bitDIO = 0x12
    PipeInDAC1 = 0x90
    PipeInDAC2 = 0x91
    PipeInDAC3 = 0x92
    PipeInDAC4 = 0x93
    PipeInDAC5 = 0x94
    PipeInDAC6 = 0x95
    PipeInDAC7 = 0x96
    PipeInDAC8 = 0x97
    PipeInDAC9 = 0x98
    PipeInDAC10 = 0x99
    PipeInDAC11 = 0x9A
    PipeInDAC12 = 0x9B
    ExpanderInfo = 0x35


class RHS(EndPoints):
    TTL_override = 0x1400
    WireInResetRun = 0x00
    WireInMaxTimeStep = 0x01
    WireInMaxTimeStepLsb = 0x01
    WireInMaxTimeStepMsb = 0x02
    WireInDataFreqPll = 0x03
    WireInMisoDelay = 0x04
    WireInStimCmdMode = 0x05
    WireInStimRegAddr = 0x06
    WireInStimRegWord = 0x07
    WireInDcAmpConvert = 0x08
    WireInExtraStates = 0x09
    WireInDacReref = 0x0a
    WireInAuxEnable = 0x0c
    WireInGlobalSettleSelect = 0x0d
    WireInAdcThreshold = 0x0f
    WireInSerialDigitalInCntl = 0x10
    WireInTtlOut32 = 0x10
    WireInLedDisplay = 0x11
    WireInManualTriggers = 0x12
    WireInTtlOutMode = 0x13
    WireInDataStreamEn = 0x14
    WireInTtlOut = 0x15
    WireInDacSource1 = 0x16
    WireInDacSource2 = 0x17
    WireInDacSource3 = 0x18
    WireInDacSource4 = 0x19
    WireInDacSource5 = 0x1a
    WireInDacSource6 = 0x1b
    WireInDacSource7 = 0x1c
    WireInDacSource8 = 0x1d
    WireInDacSource9 = 0x48
    WireInDacSource10 = 0x49
    WireInDacSource11 = 0x4A
    WireInDacSource12 = 0x4B
    WireInDacManual = 0x1e
    WireInMultiUse = 0x1f
    TrigInConfig = 0x40
    TrigInDcmProg = 0x40
    TrigInSpiStart = 0x41
    TrigInRamAddrReset = 0x42
    TrigInDacThresh = 0x43
    TrigInDacHpf = 0x44
    TrigInAuxCmdLength = 0x45
    WireOutNumWords = 0x20
    WireOutNumWordsLsb = 0x20
    WireOutNumWordsMsb = 0x21
    WireOutSpiRunning = 0x22
    WireOutTtlIn = 0x23
    WireOutDataClkLocked = 0x24
    WireOutBoardMode = 0x25
    WireOutSerialDigitalIn = 0x26
    WireOutBoardId = 0x3e
    WireOutBoardVersion = 0x3f
    PipeInAuxCmd1Msw = 0x80
    PipeInAuxCmd1 = 0x81
    PipeInAuxCmd1Lsw = 0x81
    PipeInAuxCmd2Msw = 0x82
    PipeInAuxCmd2 = 0x83
    PipeInAuxCmd2Lsw = 0x83
    PipeInAuxCmd3Msw = 0x84
    PipeInAuxCmd3 = 0x85
    PipeInAuxCmd3Lsw = 0x85
    PipeInAuxCmd4Msw = 0x86
    PipeInAuxCmd4 = 0x87
    PipeInAuxCmd4Lsw = 0x87
    PipeOutData = 0xa0
    TrigVStim = 0x40
    TrigMCU = 0x48
    WireOutXDAQStatus = 0x22
    WireInMCUControl = 0x02
    PipeInFirmware = 0x88
    PipeOutFirmware = 0xb0
    WireInSetMode = 0x00
    Enable32bitDIO = 0x0B
    PipeInDAC1 = 0x90
    PipeInDAC2 = 0x91
    PipeInDAC3 = 0x92
    PipeInDAC4 = 0x93
    PipeInDAC5 = 0x94
    PipeInDAC6 = 0x95
    PipeInDAC7 = 0x96
    PipeInDAC8 = 0x97
    PipeInDAC9 = 0x98
    PipeInDAC10 = 0x99
    PipeInDAC11 = 0x9A
    PipeInDAC12 = 0x9B
    ExpanderInfo = 0x35


class SampleRate(Enum):
    SampleRate1000Hz = (7, 125, 1000)
    SampleRate1250Hz = (7, 100, 1250)
    SampleRate1500Hz = (21, 250, 1500)
    SampleRate2000Hz = (14, 125, 2000)
    SampleRate2500Hz = (35, 250, 2500)
    SampleRate3000Hz = (21, 125, 3000)
    SampleRate3333Hz = (14, 75, 3333)
    SampleRate4000Hz = (28, 125, 4000)
    SampleRate5000Hz = (7, 25, 5000)
    SampleRate6250Hz = (7, 20, 6250)
    SampleRate8000Hz = (112, 250, 8000)
    SampleRate10000Hz = (14, 25, 10000)
    SampleRate12500Hz = (7, 10, 12500)
    SampleRate15000Hz = (21, 25, 15000)
    SampleRate20000Hz = (28, 25, 20000)
    SampleRate25000Hz = (35, 25, 25000)
    SampleRate30000Hz = (42, 25, 30000)

    @classmethod
    def fromRate(cls, rate: int):
        for sr in SampleRate:
            if sr.value[2] == rate:
                return sr
        raise ValueError('Invalid sample rate')

    @property
    def rate(self):
        return self.value[2]


class HeadstageChipID(Enum):
    NA = 0
    RHD2132 = 1
    RHD2216 = 2
    RHD2164 = 4
    RHS2116 = 32

    def num_channels(self):
        if self == HeadstageChipID.RHD2164:
            return 64
        elif self == HeadstageChipID.RHD2132:
            return 32
        elif self == HeadstageChipID.RHD2216:
            return 16
        elif self == HeadstageChipID.RHS2116:
            return 16
        else:
            return 0

    def num_channels_per_stream(self):
        if self == HeadstageChipID.RHD2164:
            return self.num_channels() // 2
        return self.num_channels()


class HeadstageChipMISOID(Enum):
    NA = 0
    MISO_A = 53
    MISO_B = 58


class TriggerEvent(Enum):
    Level = 0
    Edge = 1


class TriggerPolarity(Enum):
    Low = 0
    High = 1


class StimShape(Enum):
    Biphasic = 0
    BiphasicWithInterphaseDelay = 1
    Triphasic = 2
    Monophasic = 3


class StartPolarity(Enum):
    anodic = 0
    cathodic = 1


class StimRegister(Enum):
    Trigger = 0
    Param = 1
    EventAmpSettleOn = 2
    EventAmpSettleOff = 3
    EventStartStim = 4
    EventStimPhase2 = 5
    EventStimPhase3 = 6
    EventEndStim = 7
    EventRepeatStim = 8
    EventChargeRecovOn = 9
    EventChargeRecovOff = 10
    EventAmpSettleOnRepeat = 11
    EventAmpSettleOffRepeat = 12
    EventEnd = 13


class StimStepSize(Enum):
    StimStepSizeMin = 0
    StimStepSize10nA = 1
    StimStepSize20nA = 2
    StimStepSize50nA = 3
    StimStepSize100nA = 4
    StimStepSize200nA = 5
    StimStepSize500nA = 6
    StimStepSize1uA = 7
    StimStepSize2uA = 8
    StimStepSize5uA = 9
    StimStepSize10uA = 10
    StimStepSizeMax = 11

    @property
    def nA(self):
        return {
            StimStepSize.StimStepSizeMin: float('nan'),
            StimStepSize.StimStepSize10nA: 10,
            StimStepSize.StimStepSize20nA: 20,
            StimStepSize.StimStepSize50nA: 50,
            StimStepSize.StimStepSize100nA: 100,
            StimStepSize.StimStepSize200nA: 200,
            StimStepSize.StimStepSize500nA: 500,
            StimStepSize.StimStepSize1uA: 1000,
            StimStepSize.StimStepSize2uA: 2000,
            StimStepSize.StimStepSize5uA: 5000,
            StimStepSize.StimStepSize10uA: 10000,
            StimStepSize.StimStepSizeMax: float('nan')
        }[self]
