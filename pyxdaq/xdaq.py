import math
import time
from enum import Enum
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from dataclass_wizard import JSONWizard

from .board import Board
from .constants import XDAQMCU, HeadstageChipID, HeadstageChipMISOID, SampleRate, XDAQWireOut
from .datablock import DataBlock
from .rhd_driver import RHDDriver
from .rhs_driver import RHSDriver

class XDAQModel(Enum):
    UNKNOWN = 0
    CORE = 1
    ONE = 3

@dataclass
class XDAQInfo:
    serial: int
    hdmi: int
    fpga: int
    daio: int
    oled: int
    vido: int
    expr: int
    xprt: int

    @staticmethod
    def _parse_hdmi(hdmi):
        return 32 * (hdmi >> 24), 16 * (hdmi >> 16 & 0xff), hdmi >> 8 & 0xff

    @property
    def model(self):
        _, _, model = self._parse_hdmi(self.hdmi)
        return XDAQModel(model)

    @property
    def rhd(self):
        rhd, _, _ = self._parse_hdmi(self.hdmi)
        if self.model == XDAQModel.ONE:
            return rhd
        elif self.model == XDAQModel.CORE:
            return rhd // 2
        return 0

    @property
    def rhs(self):
        _, rhs, _ = self._parse_hdmi(self.hdmi)
        return rhs

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"""
Serial #:{self.serial:08x}
HDMI    :{self.hdmi:08x}
FPGA    :{self.fpga:08x}
DAIO    :{self.daio:08x}
OLED    :{self.oled:08x}
VIDO    :{self.vido:08x}
EXPR    :{self.expr:08x}
XPRT    :{self.xprt:08x}
"""


@dataclass
class StreamConfig(JSONWizard):
    available: bool = False
    chip: HeadstageChipID = HeadstageChipID.NA
    miso: HeadstageChipMISOID = HeadstageChipMISOID.NA
    delay: int = 0
    sid: int = None
    enabled: bool = False

    def __str__(self):
        stat = ('🟢' if self.enabled else '🛑') if self.available else '🚫'
        if not self.available:
            return f'{stat}Stream[{self.sid:02d}]-NA'
        return f'{stat}Stream[{self.sid:02d}]-{self.chip.name}:{self.miso.name} ~{self.delay}'


@dataclass
class HdmiPort(JSONWizard):
    streams: List[StreamConfig] = None
    portNumber: int = None

    def __len__(self):
        return len(self.streams)

    def __iter__(self):
        return iter(self.streams)

    @classmethod
    def fromChipInfos(cls, infos, sids, port) -> 'HdmiPort':

        def toconfig(info, sid):
            cdelay = int(info[0])
            cid = info[1]
            if cid == HeadstageChipID.RHD2132:
                return [
                    StreamConfig(available=True, chip=cid, delay=cdelay, sid=sid[0]),
                    StreamConfig(sid=sid[1])
                ]
            if cid == HeadstageChipID.RHD2164:
                sc = partial(StreamConfig, available=True, chip=cid, delay=cdelay)
                return [
                    sc(miso=HeadstageChipMISOID.MISO_A, sid=sid[0]),
                    sc(miso=HeadstageChipMISOID.MISO_B, sid=sid[1])
                ]
            if cid == HeadstageChipID.RHD2216:
                return [
                    StreamConfig(available=True, chip=cid, delay=cdelay, sid=sid[0]),
                    StreamConfig(sid=sid[1])
                ]
            if cid == HeadstageChipID.RHS2116:
                return [StreamConfig(available=True, chip=cid, delay=cdelay, sid=sid[0])]
            return [StreamConfig(sid=s) for s in sid]

        return cls([i for info, sid in zip(infos, sids) for i in toconfig(info, sid)], port)

    def __str__(self):
        out = ''
        for i, s in enumerate(self.streams):
            if i == 0:
                out += f' - {s}\n'
            elif i == len(self.streams) - 1:
                out += f' - {s}'
            elif i == len(self.streams) // 2:
                out += f'{self.portNumber:1d}  {s}\n'
            else:
                out += f'|  {s}\n'
        return out


@dataclass
class XDAQPorts(JSONWizard):
    spi_per_port: int
    chips_per_spi: int  # MOSI
    ddr: bool
    ports: List[HdmiPort] = None
    streams: List[StreamConfig] = None
    num_ports: int = 4

    @classmethod
    def default(cls, spi_per_port, chips_per_spi, ddr, num_ports=4):
        streams_per_port = spi_per_port * chips_per_spi * (2 if ddr else 1)
        ports = [
            HdmiPort(
                [
                    StreamConfig(sid=sid)
                    for sid in range(p * streams_per_port, (p + 1) * streams_per_port)
                ], p
            )
            for p in range(num_ports)
        ]
        streams = [s for p in ports for s in p.streams]
        return cls(
            spi_per_port=spi_per_port,
            chips_per_spi=chips_per_spi,
            ddr=ddr,
            ports=ports,
            streams=streams,
            num_ports=num_ports
        )

    def __len__(self):
        return len(self.ports)

    def __iter__(self):
        return iter(self.ports)

    @classmethod
    def fromChipInfos(cls, infos, spi_per_port, chips_per_spi, ddr, num_ports=4) -> 'XDAQPorts':
        infos = list(zip(*infos))
        chips_per_port = spi_per_port * chips_per_spi
        if len(infos) != chips_per_port * num_ports:
            raise ValueError(f'Expected {chips_per_port * num_ports} infos, got {len(infos)}')
        streams_per_port = spi_per_port * chips_per_spi * (2 if ddr else 1)
        ports = [
            HdmiPort.fromChipInfos(
                infos[p * chips_per_port:(p + 1) * chips_per_port],
                np.arange(p * streams_per_port,
                          (p + 1) * streams_per_port).reshape(chips_per_port,
                                                              2 if ddr else 1).tolist(), p
            ) for p in range(num_ports)
        ]
        sid2stream = [s for p in ports for s in p.streams]
        return XDAQPorts(
            spi_per_port=spi_per_port,
            chips_per_spi=chips_per_spi,
            ddr=ddr,
            ports=ports,
            streams=sid2stream
        )

    def __str__(self):
        return '\n\n'.join([str(p) for p in self.ports])

    def group_by_spi(self):
        streams_per_spi = self.chips_per_spi * (2 if self.ddr else 1)
        for spi in range(self.spi_per_port * self.num_ports):
            yield self.streams[spi * streams_per_spi:(spi + 1) * streams_per_spi]

    def group_by_port(self):
        streams_per_port = self.spi_per_port * self.chips_per_spi * (2 if self.ddr else 1)
        for port in range(self.num_ports):
            yield self.streams[port * streams_per_port:(port + 1) * streams_per_port]


class XDAQ(Board):
    xdaqinfo: XDAQInfo = None
    spiPorts: int = 4
    mode32DIO: bool = False
    ports: XDAQPorts = None
    debug: bool = False
    sampleRate: SampleRate = None

    def __init__(self, config_root, debug: bool = False):
        super().__init__(debug)
        self.config_root = Path(config_root)

    def getreg(self, sample_rate: SampleRate) -> Union[RHDDriver, RHSDriver]:
        R = RHSDriver if self.rhs else RHDDriver
        suffix = '_rhs.json' if self.rhs else '_rhd.json'
        return R(
            sample_rate,
            self.config_root.joinpath('reg' + suffix),
            self.config_root.joinpath('isa' + suffix),
        )

    def get_xdaq_status(self) -> XDAQMCU:
        value = self.GetWireOutValue(self.ep.WireOutXDAQStatus)
        testbit = lambda x: (value & x.value) == x.value
        if testbit(XDAQMCU.MCU_BUSY):
            return XDAQMCU.MCU_BUSY
        if testbit(XDAQMCU.MCU_ERROR):
            return XDAQMCU.MCU_ERROR
        if testbit(XDAQMCU.MCU_DONE):
            return XDAQMCU.MCU_DONE
        if value == 0:
            return XDAQMCU.MCU_IDLE
        return XDAQMCU.MCU_BUSY

    def detect_expander(self):
        start = time.time()
        while self.get_xdaq_status() == XDAQMCU.MCU_BUSY:
            if time.time() - start > 5:
                raise RuntimeError("Unable to detect MCU status")
            time.sleep(0.1)
        expanderBoardDetected = self.GetWireOutValue(self.ep.ExpanderInfo) != 0
        expanderBoardIdNumber = (self.GetWireOutValue(self.ep.WireOutSerialDigitalIn) >> 3) & 1
        return expanderBoardDetected, expanderBoardIdNumber

    def config_fpga(self, rhs: bool = False, bitfile: str = None) -> Tuple[int, int]:
        r = super().config_fpga(rhs, bitfile)
        self.ports = XDAQPorts.default(2, 1 if rhs else 2, False if rhs else True)
        return r

    def set32DIO(self, enable: bool):
        self.mode32DIO = enable
        self.SetWireInValue(self.ep.Enable32bitDIO, 0x04 * enable, 0x04)

    def getXDAQInfo(self):
        return XDAQInfo(
            self.GetWireOutValue(XDAQWireOut.Serial),
            self.GetWireOutValue(XDAQWireOut.Hdmi, False),
            self.GetWireOutValue(XDAQWireOut.Fpga, False),
            self.GetWireOutValue(XDAQWireOut.Daio, False),
            self.GetWireOutValue(XDAQWireOut.Oled, False),
            self.GetWireOutValue(XDAQWireOut.Vido, False),
            self.GetWireOutValue(XDAQWireOut.Expr, False),
            self.GetWireOutValue(XDAQWireOut.Xprt, False),
        )

    def reset_board(self):
        super().reset_board()
        # usb3 configuration
        self.SendMultiUse(self.ep.TrigInConfig, 9, 1024 // 4)
        self.SendMultiUse(self.ep.TrigInConfig, 10, 32)

    def run(self):
        self.ActivateTriggerIn(self.ep.TrigInSpiStart, 0)

    def is_running(self):
        return self.GetWireOutValue(self.ep.WireOutSpiRunning)

    def numWordsInFifo(self):
        return self.GetWireOutValue(self.ep.WireOutNumWords)

    def flush(self):
        raise NotImplementedError

    def selectAuxCommandBank(self, port: Union[int, str], auxCommandSlot, bank: int):
        """
        Select an auxiliary command slot (AuxCmd1, AuxCmd2, or AuxCmd3) and bank (0-15) for a particular SPI port
        (PortA - PortH) on the FPGA.
        Bank: 4 bit
                 |0123|4567|8901|2345|6789|0123|4567|8901|
        Bank for |P0  |P1  |P2  |P3  |P4  |P5  |P6  |P7  |
        """
        if self.rhs:  # why?
            return
        if auxCommandSlot < 0 or auxCommandSlot > 2:
            raise Exception("auxCommandSlot out of range")
        if bank < 0 or bank > 15:
            raise Exception("bank out of range")
        ep = [self.ep.WireInAuxCmdBank1, self.ep.WireInAuxCmdBank2,
              self.ep.WireInAuxCmdBank3][auxCommandSlot]
        if isinstance(port, str) and port == 'all':
            # repeat bank for all 8 ports, fill in same 4 bits bank for all ports
            bank = bank << 4 | bank
            bank = bank << 8 | bank
            bank = bank << 16 | bank
            self.SetWireInValue(ep, bank)
        else:
            self.SetWireInValue(ep, bank << (port * 4), 0xf << (port * 4))

    def selectAuxCommandLength(self, auxCommandSlot, loopIndex, endIndex):
        """
        Specify a command sequence length (endIndex = 0-1023) and command loop index (0-1023) for a particular
        auxiliary command slot (AuxCmd1, AuxCmd2, or AuxCmd3).
        """
        if auxCommandSlot < 0 or auxCommandSlot > (3 if self.rhs else 2):
            raise Exception("auxCommandSlot out of range")
        maxidx = 8191 if self.rhs else 1023
        if loopIndex < 0 or loopIndex > maxidx:
            raise Exception("loopIndex out of range")
        if endIndex < 0 or endIndex > maxidx:
            raise Exception("endIndex out of range")
        if self.rhs:
            self.SendMultiUse(self.ep.TrigInAuxCmdLength, auxCommandSlot + 4, loopIndex)
            self.SendMultiUse(self.ep.TrigInAuxCmdLength, auxCommandSlot, endIndex)
        else:
            self.SetWireInValue(
                self.ep.WireInAuxCmdLoop, loopIndex << (auxCommandSlot * 10), 0x000003ff <<
                (auxCommandSlot * 10), False
            )
            self.SetWireInValue(
                self.ep.WireInAuxCmdLength, endIndex << (auxCommandSlot * 10), 0x000003ff <<
                (auxCommandSlot * 10)
            )

    def _isDcmProgDone(self):
        return (self.GetWireOutValue(self.ep.WireOutDataClkLocked) & 0x0002) > 1

    def _isDataClockLocked(self):
        return (self.GetWireOutValue(self.ep.WireOutDataClkLocked) & 0x0001) > 0

    def setSampleRate(self, sample_rate: SampleRate):
        self.sampleRate = sample_rate
        while not self._isDcmProgDone():
            time.sleep(0.01)
        self.SetWireInValue(
            self.ep.WireInDataFreqPll, 256 * sample_rate.value[0] + sample_rate.value[1]
        )
        if self.rhs:
            self.ActivateTriggerIn(self.ep.TrigInDcmProg, 0)
        else:
            self.ActivateTriggerIn(self.ep.TrigInConfig, 0)
        while not self._isDataClockLocked():
            time.sleep(0.01)

    def setContinuousRunMode(self, enable: bool):
        self.SetWireInValue(self.ep.WireInResetRun, 0x02 * enable, 0x02)

    def setMaxTimeStep(self, maxTimeStep: int):
        if maxTimeStep < 0 or maxTimeStep > 2**32 - 1:
            raise Exception("maxTimeStep out of range")
        self.SetWireInValue(self.ep.WireInMaxTimeStep, maxTimeStep)

    def setCableDelay(self, port: Union[int, str], delay: int):
        if delay < 0 or delay > 15:
            raise Exception("delay out of range")

        if isinstance(port, str) and port == 'all':
            delay = delay << 4 | delay
            delay = delay << 8 | delay
            delay = delay << 16 | delay
            self.SetWireInValue(self.ep.WireInMisoDelay, delay)
        else:
            self.SetWireInValue(self.ep.WireInMisoDelay, delay << (4 * port), 0xf << (4 * port))

    @staticmethod
    def delayFromCableLength(length, sampleRate, unit):
        if unit == 'ft':
            length = length * 0.3048
        else:
            if unit != 'm':
                raise Exception("unit must be 'm' or 'ft'")
        speedOfLight = 299792458
        xilinxLvdsOutputDelay = 1.9e-9
        xilinxLvdsInputDelay = 1.4e-9
        rhd2000Delay = 9.0e-9
        misoSettleTime = 6.7e-9

        tStep = 1 / (2800 * sampleRate)
        cableVel = 0.555 * speedOfLight
        roundtriplength = 2 * length
        delay = (
            roundtriplength / cableVel
        ) + xilinxLvdsInputDelay + rhd2000Delay + xilinxLvdsOutputDelay + misoSettleTime
        delay = int(delay / tStep + 1.5)
        delay = max(delay, 1)
        return delay

    def setDspSettle(self, enable):
        self.SetWireInValue(self.ep.WireInResetRun, enable * 0x4, 0x4)

    def enableDataStream(self, stream: Union[int, str], enable: bool, force=False):
        if isinstance(stream, str) and stream == 'all':
            self.SetWireInValue(self.ep.WireInDataStreamEn, (2**32 - 1) * enable)
            # update cached value for all streams inside port object
            for s in self.ports.streams:
                s.enabled = enable
        else:
            if self.ports.streams[stream].enabled == enable:
                if not force:
                    return
            self.SetWireInValue(self.ep.WireInDataStreamEn, (0x1 * enable) << stream, 0x1 << stream)
            self.ports.streams[stream].enabled = enable

    @property
    def numDataStream(self):
        # TODO: use cached value
        return sum(i.enabled for i in self.ports.streams)

    def clearTTLout(self):
        self.setTTLout('all', False)

    def setTTLout(self, channel: Union[int, str], enable: Union[bool, int]):
        if self.rhs:
            return
        if isinstance(channel, str) and channel == 'all':
            v = enable * 0xffff if isinstance(enable, bool) else enable & 0xffff
            self.SetWireInValue(self.ep.WireInTtlOut, v)
            if self.mode32DIO:
                v = enable * 0xffff if isinstance(enable, bool) else enable >> 16
                self.SetWireInValue(self.ep.WireInTtlOut32, v)
        else:
            ep = self.ep.WireInTtlOut if channel < 16 else self.ep.WireInTtlOut32
            self.SetWireInValue(ep, int(enable) << (channel % 16), 1 << (channel % 16))

    def _getDacEndpoint(self, dacChannel):
        if dacChannel < 0 or dacChannel > 7:
            raise Exception("dacChannel out of range")
        return [
            self.ep.WireInDacSource1, self.ep.WireInDacSource2, self.ep.WireInDacSource3,
            self.ep.WireInDacSource4, self.ep.WireInDacSource5, self.ep.WireInDacSource6,
            self.ep.WireInDacSource7, self.ep.WireInDacSource8
        ][dacChannel]

    def enableDac(self, channel: int, enable: bool):
        if channel < 0 or channel > 7:
            raise Exception("channel out of range")
        bm = 0x0200 if self.rhs else 0x0800
        self.SetWireInValue(self._getDacEndpoint(channel), enable * bm, bm)

    def selectDacDataStream(self, channel: bool, stream: int):
        if channel < 0 or channel > 7:
            raise Exception("channel out of range")
        if stream < 0 or stream > (9 if self.rhs else 33):
            raise Exception("stream out of range")
        self.SetWireInValue(
            self._getDacEndpoint(channel), stream << 5, 0x1e0 if self.rhs else 0x07e0
        )

    def selectDacDataChannel(self, dacChannel: bool, dataChannel: int):
        if dacChannel < 0 or dacChannel > 7:
            raise Exception("dacChannel out of range")
        if dataChannel < 0 or dataChannel > 31:
            raise Exception("dataChannel out of range")
        self.SetWireInValue(self._getDacEndpoint(dacChannel), dataChannel, 0x001f)

    def configDac(self, channel: int, enable: bool, stream: int, dataChannel: int):
        """
        M = Max stream, 32 for RHD, 8 for RHS
        For each DAC, the user may select an amplifier channel (0~M-1) and a data stream (nominally 0~M-1).
        To enable the DAC, the DacSourceEnable bit must be set high.
        If DacSourceStream is set to M, the DAC will be controlled directly by the host computer via WireInDacManual;
        the DacSourceChannel parameter is ignored in this case.
        XDAQ extension: When source stream is set to M+1, the DAC will controlled by user defined waveform uploaded via UploadWaveform.
        """
        if channel < 0 or channel > 7:
            raise Exception("channel out of range")
        if stream < 0 or stream > (9 if self.rhs else 33):
            raise Exception("stream out of range")
        if dataChannel < 0 or dataChannel > 31:
            raise Exception("dataChannel out of range")
        # RHD [enable:1bit, stream:6bit, dataChannel:5bit]
        # RHS [enable:1bit, stream:4bit, dataChannel:5bit]
        enablebit = 0x0200 if self.rhs else 0x0800
        self.SetWireInValue(
            self._getDacEndpoint(channel), (enable * enablebit) | (stream << 5) | dataChannel,
            0x3fff if self.rhs else 0xffff
        )

    def setDacManual(self, value):
        """
        When DacSourceStream is set to 32, the DAC will be controlled directly by the host computer via WireInDacManual.
        32768 = 0V
        """
        if value < 0 or value > 65535:
            raise Exception("value out of range")
        self.SetWireInValue(self.ep.WireInDacManual, value)

    def setDacGain(self, gain: int):
        if gain < 0 or gain > 7:
            raise Exception("gain out of range")
        self.SetWireInValue(self.ep.WireInResetRun, gain << 13, 0xe000)

    def setAudioNoiseSuppress(self, noiseSuppress: int):
        if noiseSuppress < 0 or noiseSuppress > 127:
            raise Exception("noiseSuppress out of range")
        self.SetWireInValue(self.ep.WireInResetRun, noiseSuppress << 6, 0x1fc0)

    def setTTLMode(self, mode: Union[bool, List[bool]]):
        """
        Set TTL output mode for channels 0-7, True for control by FPGA.
        Only RHS can set individual TTL mode.
        """
        if self.rhs:
            if isinstance(mode, bool):
                val = 0xff * mode
            elif isinstance(mode, list) and len(mode) == 8:
                val = sum(1 << i for i, v in enumerate(mode) if v)
            else:
                raise Exception(f'invalid mode: {mode}')
            self.SetWireInValue(self.ep.WireInTtlOutMode, val, 0xff)
        else:
            if not isinstance(mode, bool):
                raise Exception(f'invalid mode: {mode}')
            self.SetWireInValue(self.ep.WireInResetRun, 8 * mode, 8)

    def setDacThreshold(self, channel: int, threshold: int, trigPolarity: bool):
        if channel < 0 or channel > 7:
            raise Exception("channel out of range")
        if threshold < 0 or threshold > 65535:
            raise Exception("threshold out of range")
        ep = self.ep.TrigInDacThresh if self.rhs else self.ep.TrigInDacConfig
        self.SendMultiUse(ep, channel, threshold)
        self.SendMultiUse(ep, channel + 8, int(trigPolarity))

    def enableExternalFastSettle(self, enable: bool):
        self.SendMultiUse(self.ep.TrigInConfig, 6, int(enable))

    def setExternalFastSettleChannel(self, channel: int):
        if channel < 0 or channel > 15:
            raise Exception("channel out of range")
        self.SendMultiUse(self.ep.TrigInConfig, 7, channel)

    def enableExternalDigOut(self, port: int, enable: bool):
        if self.rhs:
            return
        if port < 0 or port > 7:
            raise Exception("port out of range")
        self.SendMultiUse(self.ep.TrigInDacConfig, 16 + port, int(enable))

    # Select which of the TTL inputs 0-15 is used to control the auxiliary digital output
    # pin of the chips connected to a particular SPI port, if external control of auxout is enabled.
    def setExternalDigOutChannel(self, port: int, channel: int):
        if self.rhs:
            return
        if port < 0 or port > 7:
            raise Exception("port out of range")
        if channel < 0 or channel > 15:
            raise Exception("channel out of range")
        self.SendMultiUse(self.ep.TrigInDacConfig, 24 + port, channel)

    def config_dac_ref(self, enable: bool, stream: int = 0, channel: int = 0):
        if stream < 0 or stream > (7 if self.rhs else 31):
            raise Exception("stream out of range")
        if channel < 0 or channel > (15 if self.rhs else 31):
            raise Exception("channel out of range")
        if self.rhs:
            # this doesn't match the documentation, but RHX uses this implementation
            self.SetWireInValue(
                self.ep.WireInDacReref, (enable * 0x100) | (stream << 5) | channel, 0x1fff
            )
        else:
            self.SetWireInValue(
                self.ep.WireInDacReref, (enable * 0x400) | (stream << 5) | channel, 0x7fff
            )

    def enableAuxCommandsOnAllStreams(self):
        if not self.rhs:
            return
        self.SetWireInValue(self.ep.WireInAuxEnable, 0xff, 0xff)

    def setGlobalSettlePolicy(self, settle: List[bool], global_settle: bool):
        if not self.rhs:
            return
        if len(settle) != 4:
            raise Exception("settle must be a list of 4 booleans")
        v = sum(p * (1 << i) for i, p in enumerate(settle)) | (global_settle * 0x10)
        self.SetWireInValue(self.ep.WireInGlobalSettleSelect, v, 0x1f)

    def setStimCmdMode(self, enabled: bool):
        if not self.rhs:
            return
        self.SetWireInValue(self.ep.WireInStimCmdMode, enabled * 0x1, 0x1)

    def enableDcAmpConvert(self, enabled: bool):
        if not self.rhs:
            return
        self.SetWireInValue(self.ep.WireInDcAmpConvert, enabled * 0x1, 0x1)

    def setExtraStates(self, states: int):
        if not self.rhs:
            return
        self.SetWireInValue(self.ep.WireInExtraStates, states)

    def setAnalogInTriggerThreshold(self, threshold: float):
        if not self.rhs:
            return
        value = int(32768 * threshold / 10.24) + 32768
        value = max(0, min(65535, value))
        self.SetWireInValue(self.ep.WireInAdcThreshold, value)

    def programStimReg(self, stream: int, channel: int, reg: int, value: int):
        # stream(0-7) * channel(0-15) = max 128
        # WireInStimRegAddr[3:0]: StimRegAddress
        # WireInStimRegAddr[7:4]: StimRegChannel 0-15 (channel on each RHS2116)
        # WireInStimRegAddr[12:8]: StimRegModule 0-7 spi
        # WireInStimRegWord[15:0]: StimProgWord
        # 14 address
        # 0 TriggerParams
        # 1 StimParams
        # events 16 bits t + event * 1/sample_rate
        # 2 EventAmpSettleOn
        # 3 EventAmpSettleOff
        # 4 EventStartStim
        # 5 EventStimPhase2
        # 6 EventStimPhase3
        # 7 EventEndStim
        # 8 EventRepeatStim
        # 9 EventChargeRecovOn
        # 10 EventChargeRecovOff
        # 11 EventAmpSettleOnRepeat
        # 12 EventAmpSettleOffRepeat
        # 13 EventEnd
        self.SetWireInValue(
            self.ep.WireInStimRegAddr, (stream << 8) | (channel << 4) | reg, update=False
        )
        self.SetWireInValue(self.ep.WireInStimRegWord, value)
        self.ActivateTriggerIn(self.ep.TrigInRamAddrReset, 1)

    def set_headstage_sequencer(self):
        return
        raise NotImplementedError

    def initialize(self):
        self.reset_board()
        self.enableAuxCommandsOnAllStreams()
        self.setGlobalSettlePolicy([False, False, False, False], False)
        self.setSampleRate(SampleRate.SampleRate30000Hz)
        for auxCommandSlot in range(3):
            self.selectAuxCommandBank('all', auxCommandSlot, 0)
        for auxCommandSlot in range(3 + int(self.rhs)):
            self.selectAuxCommandLength(auxCommandSlot, 0, 0)
        self.setStimCmdMode(False)
        self.setContinuousRunMode(True)
        self.setMaxTimeStep(2**32 - 1)
        self.setCableDelay('all', self.delayFromCableLength(3.0, 30000, 'ft'))

        self.setDspSettle(False)
        self.enableDataStream('all', False, True)
        self.enableDataStream(0, True, True)

        self.enableDcAmpConvert(True)
        self.setExtraStates(0)
        self.clearTTLout()
        for i in range(8):
            self.configDac(i, False, 0, 0)  # Initially point DACs to DacManual1 input
        self.setDacManual(32768)
        self.setDacGain(0)
        self.setAudioNoiseSuppress(0)
        self.setTTLMode(False)

        for i in range(8):
            self.setDacThreshold(i, 32768, True)

        self.enableExternalFastSettle(False)
        self.setExternalFastSettleChannel(0)
        for i in range(8):
            self.enableExternalDigOut(i, False)
        for i in range(8):
            self.setExternalDigOutChannel(i, 0)
        self.config_dac_ref(False)
        self.enableDacHighpassFilter(False)

        self.setAnalogInTriggerThreshold(1.65)
        self.set_headstage_sequencer()

        self.xdaqinfo = self.getXDAQInfo()

    def uploadCommandList(self, commandList: np.ndarray, auxCommandSlot, bank):
        if auxCommandSlot < 0 or auxCommandSlot > (2 + int(self.rhs)):
            raise Exception("auxCommandSlot out of range")
        if bank < 0 or bank > 15:
            raise Exception("bank out of range")
        if self.rhs:
            self.ActivateTriggerIn(self.ep.TrigInRamAddrReset, 0)
            ep = [
                self.ep.PipeInAuxCmd1, self.ep.PipeInAuxCmd2, self.ep.PipeInAuxCmd3,
                self.ep.PipeInAuxCmd4
            ][auxCommandSlot]
            self.WriteToBlockPipeIn(ep, 16, bytearray(commandList.tobytes(order='C')))
        else:
            self.SetWireInValue(self.ep.WireInCmdRamBank, bank, update=False)
            for i, cmd in enumerate(commandList):
                self.SetWireInValue(self.ep.WireInCmdRamData, int(cmd), update=False)
                self.SetWireInValue(self.ep.WireInCmdRamAddr, i)
                self.ActivateTriggerIn(self.ep.TrigInConfig, auxCommandSlot + 1)

    def setDacHighpassFilter(self, cutoff: float, smapleRate: float):
        """
        Set cutoff frequency (in Hz) for optional FPGA-implemented digital high-pass filters
        associated with DAC outputs on USB interface board.  These one-pole filters can be used
        to record wideband neural data while viewing only spikes without LFPs on the DAC outputs,
        for example.  This is useful when using the low-latency FPGA thresholds to detect spikes
        and produce digital pulses on the TTL outputs, for example.
        """

        # Note that the filter coefficient is a function of the amplifier sample rate, so this
        # function should be called after the sample rate is changed.
        b = 1.0 - math.exp(-2.0 * math.pi * cutoff / smapleRate)

        # In hardware, the filter coefficient is represented as a 16-bit number.
        filterCoefficient = int(math.floor(65536.0 * b + 0.5))
        if filterCoefficient < 1:
            filterCoefficient = 1
        elif filterCoefficient > 65535:
            filterCoefficient = 65535
        self.SendMultiUse(self.ep.TrigInConfig, 5, filterCoefficient)

    def uploadCommands(self, fastSettle: bool = False, update_stim: bool = False):
        reg = self.getreg(self.sampleRate)

        if self.rhs:
            cmd = reg.dummy(8192)
            for aux in [1, 2, 3]:
                self.uploadCommandList(cmd, aux, 0)
        else:
            cmd = reg.createCommandListUpdateDigOut()
            self.uploadCommandList(cmd, 0, 0)
            self.selectAuxCommandLength(0, 0, len(cmd) - 1)
            self.selectAuxCommandBank('all', 0, 0)

            cmd = reg.createCommandListTempSensor()
            self.uploadCommandList(cmd, 1, 0)
            self.selectAuxCommandLength(1, 0, len(cmd) - 1)
            self.selectAuxCommandBank('all', 1, 0)

        # Not implemented yet
        # reg.setDspCutoffFreq(0)
        reg.set_upper_bandwidth(7500)
        if self.rhs:
            reg.set_lower_bandwidth_a(1000)
            reg.set_lower_bandwidth_b(1)
        else:
            reg.set_lower_bandwidth(1)

        if self.rhs:
            cmd = reg.createCommandListRegisterConfig(update_stim=update_stim, readonly=False)
            self.uploadCommandList(cmd, 0, 0)
            self.selectAuxCommandLength(0, 0, len(cmd) - 1)
            self.readDataBlocks()
        else:
            cmd = reg.createCommandListRegisterConfig(True)
            self.uploadCommandList(cmd, 2, 0)
            self.selectAuxCommandLength(2, 0, len(cmd) - 1)

            cmd = reg.createCommandListRegisterConfig(False)
            self.uploadCommandList(cmd, 2, 1)
            self.selectAuxCommandLength(2, 0, len(cmd) - 1)

            reg.controller.set('ampFastSettle', 1)
            cmd = reg.createCommandListRegisterConfig(False)
            self.uploadCommandList(cmd, 2, 2)
            self.selectAuxCommandLength(2, 0, len(cmd) - 1)
            reg.controller.set('ampFastSettle', 0)

            self.selectAuxCommandBank('all', 2, 2 if fastSettle else 1)

    def changeSampleRate(
        self, sampleRate: SampleRate, fastSettle: bool = False, update_stim: bool = False
    ):
        self.setSampleRate(sampleRate)

        self.setDacHighpassFilter(250, sampleRate.value[2])

        self.uploadCommands(fastSettle, update_stim)

    @staticmethod
    def _encodeWaveform(waveform):
        # 32768 = 2^15 = 0V
        # 10.24 V = 65535 = 2^16 - 1
        return np.clip(waveform * 3200 + 32768, 0, 2**16 - 1).astype('<u2').tobytes(order='C')

    def _getPipeInDACendpoint(self, dacChannel: int):
        if dacChannel < 0 or dacChannel > 7:
            raise Exception("dacChannel out of range")
        return [
            self.ep.PipeInDAC1, self.ep.PipeInDAC2, self.ep.PipeInDAC3, self.ep.PipeInDAC4,
            self.ep.PipeInDAC5, self.ep.PipeInDAC6, self.ep.PipeInDAC7, self.ep.PipeInDAC8
        ][dacChannel]

    def uploadDACData(self, waveform: np.array, dacChannel: int, length: int):
        buffer = bytearray(self._encodeWaveform(waveform))
        self.ActivateTriggerIn(self.ep.TrigInSpiStart, 2)
        result = self.WriteToBlockPipeIn(self._getPipeInDACendpoint(dacChannel), 16, buffer)
        if result < 0:
            raise Exception("Upload waveform WriteToBlockPipeIn failed")
        self.SendMultiUse(self.ep.TrigInSpiStart, 8 + dacChannel, length)

    def uploadWaveform(self, waveform, channel, length):

        self.uploadDACData(waveform, channel, length)
        self.selectDacDataStream(channel, 33)
        self.enableDac(channel, True)

    def start(self):
        self.setContinuousRunMode(True)
        self.run()

    def stop(self):
        self.setContinuousRunMode(False)
        self.setMaxTimeStep(0)
        # self.flush()

    def readDataToBuffer(self, buffer: bytearray):
        return self.ReadFromBlockPipeOut(self.ep.PipeOutData, 1024, buffer)

    def pipeoutThrottle(self, enable: bool):
        self.SetWireInValue(self.ep.WireInResetRun, int(not enable) << 16, 1 << 16)

    def disablePipeoutThrottle(self):

        class Context:

            def __init__(self, dev):
                self.dev = dev

            def __enter__(self):
                self.dev.pipeoutThrottle(False)

            def __exit__(self, exc_type, exc_value, traceback):
                self.dev.pipeoutThrottle(True)

        return Context(self)

    def discardFIFO(self):
        self.SetWireInValue(self.ep.WireInResetRun, 1 << 17, 1 << 17)
        self.SetWireInValue(self.ep.WireInResetRun, 0 << 17, 1 << 17)
        fifo = self.numWordsInFifo() * 2
        with self.disablePipeoutThrottle():
            while fifo > 0:
                tr = min(2**20, ((fifo + 1023) // 1024) * 1024)
                n = self.readDataToBuffer(bytearray(tr))
                fifo -= n

    def runAndCleanup(self):

        class Context:

            def __init__(self, dev: XDAQ):
                self.dev = dev

            def __enter__(self):
                self.dev.start()

            def __exit__(self, exc_type, exc_value, traceback):
                self.dev.stop()
                while self.dev.is_running() > 0:
                    time.sleep(0.01)
                self.dev.discardFIFO()

        return Context(self)

    def readDataBlocks(self, n=1, poll_interval: float = 0.001) -> Tuple[int, bytearray]:
        # Since our longest command sequence is 128 commands, we run the SPI interface for 128 samples.
        self.setMaxTimeStep(128 * n)
        self.setContinuousRunMode(False)
        self.run()
        while self.is_running() > 0:
            time.sleep(poll_interval * n)
        bs = self.getBlockSizeBytes() * n
        buffer = bytearray(max(((bs + 1023) // 1024) * 1024, 1024))
        r = self.readDataToBuffer(buffer)
        return (r, buffer)

    def testCableDelay(self, output: str = ''):
        headstagename = np.array([ord(i) for i in ('INTAN' if self.rhs else 'INTANRHD')])
        headstageids = np.array([i.value for i in HeadstageChipID])
        n_streams = 8 if self.rhs else 16
        if self.rhs:
            for stream in range(8):
                self.enableDataStream(stream, True)
        else:
            for stream in range(32):
                self.enableDataStream(stream, (stream % 2) == 0)
        self.selectAuxCommandBank('all', 2, 0)
        self.setMaxTimeStep(128)
        self.setContinuousRunMode(False)
        results = []
        sample_size = self.getSampleSizeBytes()
        for delay in range(16):
            self.setCableDelay('all', delay)
            self.run()
            while self.is_running() > 0:
                time.sleep(0.003)
            bs = self.getBlockSizeBytes()
            buffer = bytearray(max(((bs + 1023) // 1024) * 1024, 1024))
            self.readDataToBuffer(buffer)
            sp = DataBlock.from_buffer(
                self.rhs, sample_size, buffer, self.numDataStream, self.mode32DIO
            ).to_samples()
            nameok = (sp.device_name().T.astype(np.uint8) == headstagename).all(axis=1)
            ids, miso = sp.device_id()
            results.append(
                (
                    nameok &  # Check if the name is correct
                    (
                        np.isin(ids, headstageids) &  # Check if the ID is correct
                        (
                            (ids != HeadstageChipID.RHD2164.value) |
                            (miso == HeadstageChipMISOID.MISO_A.value)
                        )  # Check MISO for RHD2164
                    ),
                    ids,
                    miso,
                )
            )
        # Find the first delay that has a valid result
        delay = np.argmax(np.stack([i[0] for i in results]), axis=0)
        # Delay x (Stream x (3,) ) -> Delay x 3 x Stream
        results = np.array(results, dtype=int).transpose(0, 2, 1).tolist()

        def cast(valid: int, cid: int, miso: int) -> Tuple[HeadstageChipID, HeadstageChipMISOID]:
            if valid == 0:
                return (HeadstageChipID.NA, HeadstageChipMISOID.NA)
            return (HeadstageChipID(cid), HeadstageChipMISOID(miso))

        if output == 'all':
            return [[cast(*r) for r in res_delay] for res_delay in results]
        # lookup the detected chip at valid delay for each non ddr stream
        return delay, *zip(*[cast(*results[delay[s]][s]) for s in range(n_streams)])

    def findConnectedAmplifiers(self):
        for sampleRate in sorted(list(SampleRate), key=lambda x: x.value[2], reverse=True):
            self.changeSampleRate(sampleRate)
            self.ports = XDAQPorts.fromChipInfos(
                self.testCableDelay(), 2, 1 if self.rhs else 2, not self.rhs
            )
            for spi, streams in enumerate(
                    self.ports.group_by_port() if self.rhs else self.ports.group_by_spi()):
                self.setCableDelay(spi, max(s.delay for s in streams))
            # since ports are replaced, we need a force update to sync the enable state
            # it could be done by copying the cached state from the old ports
            self.enableDataStream('all', False, True)
            for s in self.ports.streams:
                self.enableDataStream(s.sid, s.available)
            self.setSpiLedDisplay([any(c.available for c in p) for p in self.ports])
            return

    def setSpiLedDisplay(self, stat: List[bool]):
        value = sum(1 << i for i, v in enumerate(stat) if v)
        self.SendMultiUse(self.ep.TrigInConfig, 8, value)

    def enableDacHighpassFilter(self, enable: bool):
        self.SendMultiUse(self.ep.TrigInConfig, 4, 1 if enable else 0)

    def getBlockSizeBytes(self):
        return getBlocksizeInWords(self.rhs, self.mode32DIO, 128, self.numDataStream, 32) * 2

    def getSampleSizeBytes(self):
        return getBlocksizeInWords(self.rhs, self.mode32DIO, 1, self.numDataStream, 32) * 2

    def getSampleRate(self) -> int:
        return self.sampleRate.value[2]

    def calibrateADC(self, fastSettle: bool = False):
        # RHD: Select RAM Bank 0 for AuxCmd3 initially, so the ADC is calibrated.
        # RHS: use CLEAR command to calibrate ADC
        self.selectAuxCommandBank('all', 2, 0)
        self.readDataBlocks(poll_interval=0.003)  # upload aux commands
        self.selectAuxCommandBank('all', 2, 2 if fastSettle else 1)


def get_XDAQ(
    *,
    rhs: bool = False,
    bitfile: str = None,
    config_root: str = 'config',
    fastSettle: bool = False,
    skip_headstage: bool = False,
    debug: bool = False
):
    xdaq = XDAQ(config_root, debug=debug)
    xdaq.config_fpga(rhs, bitfile)
    xdaq.initialize()

    xdaq.changeSampleRate(SampleRate.SampleRate30000Hz, fastSettle)
    if skip_headstage:
        return xdaq

    xdaq.findConnectedAmplifiers()
    xdaq.calibrateADC(fastSettle)
    xdaq.changeSampleRate(SampleRate.SampleRate20000Hz, fastSettle)
    return xdaq


def getBlocksizeInWords(rhs, mode32DIO, samplesPerDataBlock, numDataStreams, channelsPerStream):
    if rhs:
        # 4 = magic number; 2 = time stamp; 20 = (16 amp channels + 4 aux commands, each 32 bit results);
        # 4 = stim control params; 8 = DACs; 8 = ADCs; 2 = TTL in/out
        sampleSize = 4 + 2 + numDataStreams * (2 * (16 + 4) + 4) + 8 + 8 + 2
        return samplesPerDataBlock * sampleSize
    else:
        # 4 = magic number; 2 = time stamp; 35 = (32 amp channels + 3 aux commands); 0-3 filler words; 8 = ADCs; 4 = TTL in/out
        padding = ((numDataStreams + 2 * mode32DIO) % 4)
        sampleSize = (
            4 + 2 + (numDataStreams * (channelsPerStream + 3)) + padding + 8 + 2 + 2 * mode32DIO
        )
        return samplesPerDataBlock * sampleSize
