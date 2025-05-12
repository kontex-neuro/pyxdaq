import math
import time
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import List, Tuple, Union

import numpy as np
from dataclass_wizard import JSONWizard
from tqdm.auto import tqdm

from . import impedance, resources
from .board import Board
from .constants import *
from .datablock import DataBlock
from .rhd_driver import RHDDriver
from .rhs_driver import RHSDriver


class XDAQModel(Enum):
    NA = 0
    One = 3
    Core = 1


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

    def group_by_chip(self):
        streams_per_chip = (2 if self.ddr else 1)
        for chip in range(self.spi_per_port * self.chips_per_spi * self.num_ports):
            yield self.streams[chip * streams_per_chip:(chip + 1) * streams_per_chip]


def stim_trigger(source: int, event: TriggerEvent, polarity: TriggerPolarity, enabled: bool):
    return source | event.value << 5 | polarity.value << 6 | enabled << 7


def stim_params(pulses: int, shape: StimShape, start_polarity: StartPolarity):
    return (pulses - 1) | shape.value << 8 | start_polarity.value << 10


class XDAQ:
    dev: Board
    expander: bool = None
    spiPorts: int = 4
    ports: XDAQPorts = None
    debug: bool = False
    sampleRate: SampleRate = None
    rhs: bool = None
    ep: Union[RHD, RHS, None] = None

    def __init__(self, dev: Board):
        self.dev = dev
        self.ep = RHS if self.dev.rhs else RHD
        self.rhs = self.dev.rhs
        self.expander = self.detect_expander()
        self.ports = XDAQPorts.default(2, 1 if self.dev.rhs else 2, False if self.dev.rhs else True)

    def getreg(self, sample_rate: SampleRate) -> Union[RHDDriver, RHSDriver]:
        R = RHSDriver if self.rhs else RHDDriver
        res = resources.rhs if self.rhs else resources.rhd
        return R(sample_rate, res.reg_path, res.isa_path)

    def detect_expander(self):
        expanderBoardDetected = self.dev.GetWireOutValue(self.ep.ExpanderInfo) != 0
        expanderBoardIdNumber = (self.dev.GetWireOutValue(self.ep.WireOutSerialDigitalIn) >> 3) & 1
        return expanderBoardDetected, expanderBoardIdNumber

    def reset_board(self):
        """
        This clears all auxiliary command RAM banks, clears the USB FIFO, and resets the
        per-channel sampling rate to 30.0 kS/s/ch.
        """
        self.dev.SetWireInValue(self.ep.WireInResetRun, 1, 1)
        self.dev.SetWireInValue(self.ep.WireInResetRun, 0, 1)

    def run(self):
        self.dev.ActivateTriggerIn(self.ep.TrigInSpiStart, 0)

    def resetSequencers(self):
        self.dev.ActivateTriggerIn(self.ep.TrigInSpiStart, 1)

    def is_running(self):
        return self.dev.GetWireOutValue(self.ep.WireOutSpiRunning) & 1

    def flush(self):
        self.dev.raw.read(0xA0, bytearray())

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
            self.dev.SetWireInValue(ep, bank)
        else:
            self.dev.SetWireInValue(ep, bank << (port * 4), 0xf << (port * 4))

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
            self.dev.SendTrig(
                self.ep.TrigInAuxCmdLength, auxCommandSlot + 4, self.ep.WireInMultiUse, loopIndex
            )
            self.dev.SendTrig(
                self.ep.TrigInAuxCmdLength, auxCommandSlot, self.ep.WireInMultiUse, endIndex
            )
        else:
            self.dev.SetWireInValue(
                self.ep.WireInAuxCmdLoop, loopIndex << (auxCommandSlot * 10), 0x000003ff <<
                (auxCommandSlot * 10), False
            )
            self.dev.SetWireInValue(
                self.ep.WireInAuxCmdLength, endIndex << (auxCommandSlot * 10), 0x000003ff <<
                (auxCommandSlot * 10)
            )

    def _isDcmProgDone(self):
        return (self.dev.GetWireOutValue(self.ep.WireOutDataClkLocked) & 0x0002) > 1

    def _isDataClockLocked(self):
        return (self.dev.GetWireOutValue(self.ep.WireOutDataClkLocked) & 0x0001) > 0

    def setSampleRate(self, sample_rate: SampleRate):
        self.sampleRate = sample_rate
        while not self._isDcmProgDone():
            time.sleep(0.01)
        self.dev.SetWireInValue(
            self.ep.WireInDataFreqPll, 256 * sample_rate.value[0] + sample_rate.value[1]
        )
        if self.rhs:
            self.dev.ActivateTriggerIn(self.ep.TrigInDcmProg, 0)
        else:
            self.dev.ActivateTriggerIn(self.ep.TrigInConfig, 0)
        while not self._isDataClockLocked():
            time.sleep(0.01)

    def setContinuousRunMode(self, enable: bool):
        self.dev.SetWireInValue(self.ep.WireInResetRun, 0x02 * enable, 0x02)

    def setMaxTimeStep(self, maxTimeStep: int):
        if maxTimeStep < 0 or maxTimeStep > 2**32 - 1:
            raise Exception("maxTimeStep out of range")
        self.dev.SetWireInValue(self.ep.WireInMaxTimeStep, maxTimeStep)

    def setCableDelay(self, port: Union[int, str], delay: int):
        if delay < 0 or delay > 15:
            raise Exception("delay out of range")
        if (isinstance(port, str) and port != 'all') or (isinstance(port, int) and
                                                         (port < 0 or port > 7)):
            raise Exception("port out of range")

        if isinstance(port, str) and port == 'all':
            delay = delay << 4 | delay
            delay = delay << 8 | delay
            delay = delay << 16 | delay
            self.dev.SetWireInValue(self.ep.WireInMisoDelay, delay)
        else:
            self.dev.SetWireInValue(self.ep.WireInMisoDelay, delay << (4 * port), 0xf << (4 * port))

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
        self.dev.SetWireInValue(self.ep.WireInResetRun, enable * 0x4, 0x4)

    def enableDataStream(self, stream: Union[int, str], enable: bool, force=False):
        if isinstance(stream, str) and stream == 'all':
            self.dev.SetWireInValue(self.ep.WireInDataStreamEn, (2**32 - 1) * enable)
            # update cached value for all streams inside port object
            for s in self.ports.streams:
                s.enabled = enable
        else:
            if self.ports.streams[stream].enabled == enable:
                if not force:
                    return
            self.dev.SetWireInValue(
                self.ep.WireInDataStreamEn, (0x1 * enable) << stream, 0x1 << stream
            )
            self.ports.streams[stream].enabled = enable

    @property
    def numDataStream(self):
        # TODO: use cached value
        return sum(i.enabled for i in self.ports.streams)

    def set_ttl_override(self, enable: Union[int, bool]):
        v = enable * 0xffffffff if isinstance(enable, bool) else enable & 0xffffffff
        self.dev.SetWireInValue(self.ep.TTL_override, v)

    def setTTLout(self, channel: Union[int, str], enable: Union[bool, int]):
        if isinstance(channel, str) and channel == 'all':
            v = enable * 0xffffffff if isinstance(enable, bool) else enable & 0xffffffff
            self.dev.SetWireInValue(self.ep.WireInTtlOut, v)
        else:
            self.dev.SetWireInValue(self.ep.WireInTtlOut, int(enable) << channel, 1 << channel)

    def _getDacEndpoint(self, dacChannel):
        if dacChannel < 0 or dacChannel > 11:
            raise ValueError("dacChannel out of range")
        return [
            self.ep.WireInDacSource1, self.ep.WireInDacSource2, self.ep.WireInDacSource3,
            self.ep.WireInDacSource4, self.ep.WireInDacSource5, self.ep.WireInDacSource6,
            self.ep.WireInDacSource7, self.ep.WireInDacSource8, self.ep.WireInDacSource9,
            self.ep.WireInDacSource10, self.ep.WireInDacSource11, self.ep.WireInDacSource12
        ][dacChannel]

    def enableDac(self, channel: int, enable: bool):
        if channel < 0 or channel > 11:
            raise ValueError("channel out of range")
        bm = 0x0200 if self.rhs else 0x0800
        self.dev.SetWireInValue(self._getDacEndpoint(channel), enable * bm, bm)

    def selectDacDataStream(self, channel: bool, stream: int):
        if channel < 0 or channel > 11:
            raise ValueError("channel out of range")
        if stream < 0 or stream > (9 if self.rhs else 33):
            raise ValueError("stream out of range")
        self.dev.SetWireInValue(
            self._getDacEndpoint(channel), stream << 5, 0x1e0 if self.rhs else 0x07e0
        )

    def selectDacDataChannel(self, dacChannel: bool, dataChannel: int):
        if dacChannel < 0 or dacChannel > 7:
            raise ValueError("dacChannel out of range")
        if dataChannel < 0 or dataChannel > 31:
            raise ValueError("dataChannel out of range")
        self.dev.SetWireInValue(self._getDacEndpoint(dacChannel), dataChannel, 0x001f)

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
        self.dev.SetWireInValue(
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
        self.dev.SetWireInValue(self.ep.WireInDacManual, value)

    def setDacGain(self, gain: int):
        if gain < 0 or gain > 7:
            raise Exception("gain out of range")
        self.dev.SetWireInValue(self.ep.WireInResetRun, gain << 13, 0xe000)

    def setAudioNoiseSuppress(self, noiseSuppress: int):
        if noiseSuppress < 0 or noiseSuppress > 127:
            raise Exception("noiseSuppress out of range")
        self.dev.SetWireInValue(self.ep.WireInResetRun, noiseSuppress << 6, 0x1fc0)

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
            self.dev.SetWireInValue(self.ep.WireInTtlOutMode, val, 0xff)
        else:
            if not isinstance(mode, bool):
                raise Exception(f'invalid mode: {mode}')
            self.dev.SetWireInValue(self.ep.WireInResetRun, 8 * mode, 8)

    def setDacThreshold(self, channel: int, threshold: int, trigPolarity: bool):
        if channel < 0 or channel > 7:
            raise Exception("channel out of range")
        if threshold < 0 or threshold > 65535:
            raise Exception("threshold out of range")
        ep = self.ep.TrigInDacThresh if self.rhs else self.ep.TrigInDacConfig
        self.dev.SendTrig(ep, channel, self.ep.WireInMultiUse, threshold)
        self.dev.SendTrig(ep, channel + 8, self.ep.WireInMultiUse, int(trigPolarity))

    def enableExternalFastSettle(self, enable: bool):
        self.dev.SendTrig(self.ep.TrigInConfig, 6, self.ep.WireInMultiUse, int(enable))

    def setExternalFastSettleChannel(self, channel: int):
        if channel < 0 or channel > 15:
            raise Exception("channel out of range")
        self.dev.SendTrig(self.ep.TrigInConfig, 7, self.ep.WireInMultiUse, channel)

    def enableExternalDigOut(self, port: int, enable: bool):
        if self.rhs:
            return
        if port < 0 or port > 7:
            raise Exception("port out of range")
        self.dev.SendTrig(self.ep.TrigInDacConfig, 16 + port, self.ep.WireInMultiUse, int(enable))

    # Select which of the TTL inputs 0-15 is used to control the auxiliary digital output
    # pin of the chips connected to a particular SPI port, if external control of auxout is enabled.
    def setExternalDigOutChannel(self, port: int, channel: int):
        if self.rhs:
            return
        if port < 0 or port > 7:
            raise Exception("port out of range")
        if channel < 0 or channel > 15:
            raise Exception("channel out of range")
        self.dev.SendTrig(self.ep.TrigInDacConfig, 24 + port, self.ep.WireInMultiUse, channel)

    def config_dac_ref(self, enable: bool, stream: int = 0, channel: int = 0):
        if stream < 0 or stream > (7 if self.rhs else 31):
            raise Exception("stream out of range")
        if channel < 0 or channel > (15 if self.rhs else 31):
            raise Exception("channel out of range")
        if self.rhs:
            # this doesn't match the documentation, but RHX uses this implementation
            self.dev.SetWireInValue(
                self.ep.WireInDacReref, (enable * 0x100) | (stream << 5) | channel, 0x1fff
            )
        else:
            self.dev.SetWireInValue(
                self.ep.WireInDacReref, (enable * 0x400) | (stream << 5) | channel, 0x7fff
            )

    def enableAuxCommandsOnAllStreams(self):
        if not self.rhs:
            return
        self.dev.SetWireInValue(self.ep.WireInAuxEnable, 0xff, 0xff)

    def enableAuxCommandsOnOneStream(self, stream):
        if not self.rhs:
            return
        self.dev.SetWireInValue(self.ep.WireInAuxEnable, 1 << stream, 0xff)

    def setGlobalSettlePolicy(self, settle: List[bool], global_settle: bool):
        if not self.rhs:
            return
        if len(settle) != 4:
            raise Exception("settle must be a list of 4 booleans")
        v = sum(p * (1 << i) for i, p in enumerate(settle)) | (global_settle * 0x10)
        self.dev.SetWireInValue(self.ep.WireInGlobalSettleSelect, v, 0x1f)

    def setStimCmdMode(self, enabled: bool):
        """
        Enable or disable stimulation for XDAQ
        """
        if not self.rhs:
            return
        self.dev.SetWireInValue(self.ep.WireInStimCmdMode, enabled * 0x1, 0x1)

    def enableDcAmpConvert(self, enabled: bool):
        if not self.rhs:
            return
        self.dev.SetWireInValue(self.ep.WireInDcAmpConvert, enabled * 0x1, 0x1)

    def setExtraStates(self, states: int):
        if not self.rhs:
            return
        self.dev.SetWireInValue(self.ep.WireInExtraStates, states)

    def setAnalogInTriggerThreshold(self, threshold: float):
        if not self.rhs:
            return
        value = int(32768 * threshold / 10.24) + 32768
        value = max(0, min(65535, value))
        self.dev.SetWireInValue(self.ep.WireInAdcThreshold, value)

    def programStimReg(self, stream: int, channel: int, reg: StimRegister, value: int):
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
        self.dev.SetWireInValue(
            self.ep.WireInStimRegAddr, (stream << 8) | (channel << 4) | reg.value, update=False
        )
        self.dev.SetWireInValue(self.ep.WireInStimRegWord, value)
        self.dev.ActivateTriggerIn(self.ep.TrigInRamAddrReset, 1)

    def set_headstage_sequencer(self):
        NEVER = 0xffff
        for stream in range(8):
            for channel in range(16):
                self.programStimReg(
                    stream, channel, StimRegister.Trigger,
                    stim_trigger(0, TriggerEvent.Edge, TriggerPolarity.Low, False)
                )
                self.programStimReg(
                    stream, channel, StimRegister.Param,
                    stim_params(1, StimShape.Biphasic, StartPolarity.cathodic)
                )
                self.programStimReg(stream, channel, StimRegister.EventAmpSettleOn, NEVER)
                self.programStimReg(stream, channel, StimRegister.EventStartStim, NEVER)
                self.programStimReg(stream, channel, StimRegister.EventStimPhase2, NEVER)
                self.programStimReg(stream, channel, StimRegister.EventStimPhase3, NEVER)
                self.programStimReg(stream, channel, StimRegister.EventEndStim, NEVER)
                self.programStimReg(stream, channel, StimRegister.EventRepeatStim, NEVER)
                self.programStimReg(stream, channel, StimRegister.EventAmpSettleOff, NEVER)
                self.programStimReg(stream, channel, StimRegister.EventChargeRecovOn, NEVER)
                self.programStimReg(stream, channel, StimRegister.EventChargeRecovOff, NEVER)
                self.programStimReg(stream, channel, StimRegister.EventAmpSettleOnRepeat, NEVER)
                self.programStimReg(stream, channel, StimRegister.EventAmpSettleOffRepeat, NEVER)
                self.programStimReg(stream, channel, StimRegister.EventEnd, 65534)

        for stream in range(8, 16):
            self.programStimReg(
                stream, 0, StimRegister.Trigger,
                stim_trigger(0, TriggerEvent.Edge, TriggerPolarity.Low, False)
            )
            self.programStimReg(
                stream, 0, StimRegister.Param,
                stim_params(1, StimShape.Monophasic, StartPolarity.cathodic)
            )
            self.programStimReg(stream, 0, StimRegister.EventStartStim, 0)
            self.programStimReg(stream, 0, StimRegister.EventStimPhase2, NEVER)
            self.programStimReg(stream, 0, StimRegister.EventStimPhase3, NEVER)
            self.programStimReg(stream, 0, StimRegister.EventEndStim, 200)
            self.programStimReg(stream, 0, StimRegister.EventRepeatStim, NEVER)
            self.programStimReg(stream, 0, StimRegister.EventEnd, 240)
            self.programStimReg(stream, 0, StimRegister.EventChargeRecovOn, 32768)
            self.programStimReg(stream, 0, StimRegister.EventChargeRecovOff, 32768 + 3200)
            self.programStimReg(stream, 0, StimRegister.EventAmpSettleOnRepeat, 32768 - 3200)

        for channel in range(16):
            self.programStimReg(
                16, channel, StimRegister.Trigger,
                stim_trigger(0, TriggerEvent.Edge, TriggerPolarity.Low, False)
            )
            self.programStimReg(
                16, channel, StimRegister.Param,
                stim_params(3, StimShape.Biphasic, StartPolarity.cathodic)
            )
            self.programStimReg(16, channel, StimRegister.EventStartStim, NEVER)
            self.programStimReg(16, channel, StimRegister.EventEndStim, NEVER)
            self.programStimReg(16, channel, StimRegister.EventRepeatStim, NEVER)
            self.programStimReg(16, channel, StimRegister.EventEnd, 65534)

    def initialize(self):
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
        self.setTTLout('all', False)
        for i in range(8):
            self.configDac(i, False, 0, 0)  # Initially point DACs to DacManual1 input
        self.setDacManual(32768)
        self.setDacGain(0)
        self.setAudioNoiseSuppress(0)
        self.setTTLMode(False)

        for i in range(8):
            self.setDacThreshold(i, 32768, True)
        if not self.rhs:
            self.enableExternalFastSettle(False)
            self.setExternalFastSettleChannel(0)
        for i in range(8):
            self.enableExternalDigOut(i, False)
        for i in range(8):
            self.setExternalDigOutChannel(i, 0)
        self.config_dac_ref(False)
        self.enableDacHighpassFilter(False)

        self.setAnalogInTriggerThreshold(1.65)
        if self.rhs:
            self.set_headstage_sequencer()

    def uploadCommandList(self, commandList: np.ndarray, auxCommandSlot, bank):
        if auxCommandSlot < 0 or auxCommandSlot > (2 + int(self.rhs)):
            raise Exception("auxCommandSlot out of range")
        if bank < 0 or bank > 15:
            raise Exception("bank out of range")
        if self.rhs:
            self.dev.ActivateTriggerIn(self.ep.TrigInRamAddrReset, 0)
            ep = [
                self.ep.PipeInAuxCmd1, self.ep.PipeInAuxCmd2, self.ep.PipeInAuxCmd3,
                self.ep.PipeInAuxCmd4
            ][auxCommandSlot]
            commandList = np.pad(commandList, (0, 16 - len(commandList) % 16), 'constant')
            self.dev.WriteToBlockPipeIn(ep, bytearray(commandList.tobytes(order='C')))
        else:
            self.dev.SetWireInValue(self.ep.WireInCmdRamBank, bank)
            for i, cmd in enumerate(commandList):
                self.dev.SetWireInValue(self.ep.WireInCmdRamData, int(cmd), update=False)
                self.dev.SetWireInValue(self.ep.WireInCmdRamAddr, i)
                self.dev.ActivateTriggerIn(self.ep.TrigInConfig, auxCommandSlot + 1)

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
        self.dev.SendTrig(self.ep.TrigInConfig, 5, self.ep.WireInMultiUse, filterCoefficient)

    def uploadCommands(
        self,
        fastSettle: bool = False,
        stim_params: dict = {
            'update_stim': False,
            'readonly': True
        },
        upper_bandwidth: float = 7500,
        lower_bandwidth: float = 1
    ):
        reg = self.getreg(self.sampleRate)

        if not self.rhs:
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
        reg.set_upper_bandwidth(upper_bandwidth)
        if self.rhs:
            reg.set_lower_bandwidth_a(1000)
            reg.set_lower_bandwidth_b(lower_bandwidth)
        else:
            reg.set_lower_bandwidth(lower_bandwidth)

        if self.rhs:
            cmd = reg.createCommandListRegisterConfig(**stim_params)
            self.uploadCommandList(cmd, 0, 0)
            self.selectAuxCommandLength(0, 0, len(cmd) - 1)
            cmd = reg.dummy(8192)
            for aux in [1, 2, 3]:
                self.uploadCommandList(cmd, aux, 0)
            cmd = reg.createCommandListRegisterConfig(update_stim=True, readonly=False)
            self.uploadCommandList(cmd, 0, 0)
            self.selectAuxCommandLength(0, 0, len(cmd) - 1)

            self.runAndReadBuffer(samples=128)
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
        self,
        sampleRate: SampleRate,
        fastSettle: bool = False,
        update_stim: bool = False,
        upper_bandwidth: float = 7500,
        lower_bandwidth: float = 1
    ):
        self.setSampleRate(sampleRate)
        if not self.rhs:
            self.setDacHighpassFilter(250, sampleRate.rate)

        self.uploadCommands(
            fastSettle, upper_bandwidth=upper_bandwidth, lower_bandwidth=lower_bandwidth
        )

    @staticmethod
    def _encodeWaveform(waveform, from_voltage: bool = True):
        # 32768 = 2^15 = 0V
        # 10.24 V = 65535 = 2^16 - 1
        if from_voltage:
            return np.clip(waveform * 3200 + 32768, 0, 2**16 - 1).astype('<u2').tobytes(order='C')
        else:
            return np.clip(waveform, 0, 2**16 - 1).astype('<u2').tobytes(order='C')

    def _getPipeInDACendpoint(self, dacChannel: int):
        if dacChannel < 0 or dacChannel > 11:
            raise RuntimeError("dacChannel out of range")
        return [
            self.ep.PipeInDAC1, self.ep.PipeInDAC2, self.ep.PipeInDAC3, self.ep.PipeInDAC4,
            self.ep.PipeInDAC5, self.ep.PipeInDAC6, self.ep.PipeInDAC7, self.ep.PipeInDAC8,
            self.ep.PipeInDAC9, self.ep.PipeInDAC10, self.ep.PipeInDAC11, self.ep.PipeInDAC12
        ][dacChannel]

    def uploadDACData(
        self, waveform: np.array, dacChannel: int, length: int, from_voltage: bool = True
    ):
        buffer = bytearray(self._encodeWaveform(waveform, from_voltage))
        self.dev.ActivateTriggerIn(self.ep.TrigInSpiStart, 2)
        result = self.dev.WriteToBlockPipeIn(self._getPipeInDACendpoint(dacChannel), buffer)
        if result < 0:
            raise RuntimeError("Upload waveform WriteToBlockPipeIn failed")
        if dacChannel <= 7:
            self.dev.SendTrig(
                self.ep.TrigInSpiStart, 8 + dacChannel, self.ep.WireInMultiUse, length
            )

    def uploadWaveform(self, waveform, channel: int, length: int, from_voltage: bool = True):

        self.uploadDACData(waveform, channel, length, from_voltage)
        self.selectDacDataStream(channel, 9 if self.rhs else 33)
        self.enableDac(channel, True)

    def start(self, *, continuous: bool = None):
        if continuous is not None:
            self.setContinuousRunMode(continuous)
        self.run()

    def stop(self, *, wait: bool = False):
        self.setMaxTimeStep(0)
        self.setContinuousRunMode(False)
        if wait:
            start = time.time()
            while self.is_running():
                time.sleep(0.01)
                if time.time() - start > 1:
                    raise TimeoutError("Timeout waiting for stop")

    def readDataToBuffer(self, buffer: bytearray):
        return self.dev.ReadFromBlockPipeOut(self.ep.PipeOutData, 1024, buffer)

    def runAndReadBuffer(self, samples) -> Tuple[int, bytearray]:
        buffers = []

        sample_size = self.getSampleSizeBytes()
        total_size = samples * sample_size

        def callback(data, error):
            if error is not None:
                print(error, sum(map(len, buffers)), total_size, self.is_running())
            if data is not None:
                buffers.append(bytearray(data))

        self.setMaxTimeStep(0)
        self.setContinuousRunMode(True)
        sample_time = samples / self.getSampleRate()
        hw_events_per_sec = 100
        expected_data_rate = self.sampleRate.rate * sample_size
        chunk_size = expected_data_rate / hw_events_per_sec
        chunk_size = 2**(int(min(max(chunk_size, 2**10), 2**20)).bit_length() - 1)

        with self.dev.start_receiving_aligned_buffer(
                self.ep.PipeOutData,
                sample_size,
                callback,
                chunk_size=chunk_size,
        ):
            self.run()
            start = time.time()
            while sum(map(len, buffers)) < total_size:
                if time.time() - start > (1 + sample_time):
                    raise TimeoutError("Timeout getting samples")
                time.sleep(0.01)
            self.setContinuousRunMode(False)
        self.flush()

        return b''.join(buffers)[:total_size]

    def runAndReadDataBlock(self, samples) -> DataBlock:
        buffer = self.runAndReadBuffer(samples)
        return DataBlock.from_buffer(
            self.rhs, self.getSampleSizeBytes(), buffer, self.numDataStream
        )

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
        results = []
        for delay in range(16):
            self.setCableDelay('all', delay)
            sp = self.runAndReadDataBlock(128).to_samples()
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
        self.selectAuxCommandBank('all', 2, 1)
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

    def setSpiLedDisplay(self, stat: List[bool]):
        value = sum(1 << i for i, v in enumerate(stat) if v)
        self.dev.SendTrig(self.ep.TrigInConfig, 8, self.ep.WireInMultiUse, value)

    def enableDacHighpassFilter(self, enable: bool):
        self.dev.SendTrig(self.ep.TrigInConfig, 4, self.ep.WireInMultiUse, 1 if enable else 0)

    def getSampleSizeBytes(self):
        return getBlocksizeInWords(self.rhs, self.numDataStream) * 2

    def getSampleRate(self) -> int:
        return self.sampleRate.value[2]

    def calibrateADC(self, fastSettle: bool = False):
        # RHD: Select RAM Bank 0 for AuxCmd3 initially, so the ADC is calibrated.
        # RHS: use CLEAR command to calibrate ADC
        self.selectAuxCommandBank('all', 2, 0)
        self.runAndReadBuffer(samples=128)  # upload aux commands
        self.selectAuxCommandBank('all', 2, 2 if fastSettle else 1)

    def manual_trigger(self, trigger: int, enable: bool):
        """
        Turn on or off manual triggers for a particular trigger event.
        """
        if not self.rhs:
            return
        self.dev.SetWireInValue(RHS.WireInManualTriggers, int(enable) << trigger, 1 << trigger)

    def set_stim(
        self, stream: int, channel: int, polarity: StartPolarity, shape: StimShape, delay_ms: float,
        duration_phase1_ms: float, duration_phase2_ms: float, duration_phase3_ms: float,
        amp_neg_mA: float, amp_pos_mA: float, pulses: int, duration_pulse_ms: float,
        pre_ampsettle_ms: float, post_ampsettle_ms: float, trigger: TriggerEvent,
        trigger_source: int, trigger_pol: TriggerPolarity, step_size: StimStepSize, enable: bool,
        post_charge_recovery_ms: float
    ):
        dt = 1 / self.sampleRate.value[2]
        if not enable:
            amp_neg_mA = 0
            amp_pos_mA = 0
        self.programStimReg(
            stream, channel, StimRegister.Trigger,
            stim_trigger(trigger_source, trigger, trigger_pol, enable)
        )
        self.programStimReg(
            stream, channel, StimRegister.Param, stim_params(pulses, shape, polarity)
        )
        t0 = int(delay_ms * 1e-3 / dt)
        t1 = int(duration_phase1_ms * 1e-3 / dt) + t0
        t2 = int(duration_phase2_ms * 1e-3 / dt) + t1
        t3 = int(duration_phase3_ms * 1e-3 / dt) + t2
        t4 = int(duration_pulse_ms * 1e-3 / dt) + t3
        if pre_ampsettle_ms > 0:
            t_ampsettle_on = t0 - int(pre_ampsettle_ms * 1e-3 / dt)
        else:
            t_ampsettle_on = t0
        if post_ampsettle_ms > 0:
            t_ampsettle_off = int(post_ampsettle_ms * 1e-3 / dt) + t3
        else:
            t_ampsettle_off = t3

        if post_charge_recovery_ms > 0:
            t_charge_recovery_on = t3
            t_charge_recovery_off = int(post_charge_recovery_ms * 1e-3 / dt) + t_charge_recovery_on
        else:
            t_charge_recovery_on = 0xFFFF
            t_charge_recovery_off = 0xFFFF

        self.programStimReg(stream, channel, StimRegister.EventAmpSettleOn, t_ampsettle_on)
        self.programStimReg(stream, channel, StimRegister.EventStartStim, t0)
        self.programStimReg(stream, channel, StimRegister.EventStimPhase2, t1)
        self.programStimReg(stream, channel, StimRegister.EventStimPhase3, t2)
        self.programStimReg(stream, channel, StimRegister.EventEndStim, t3)
        self.programStimReg(stream, channel, StimRegister.EventRepeatStim, t4)
        self.programStimReg(stream, channel, StimRegister.EventAmpSettleOff, t_ampsettle_off)
        self.programStimReg(stream, channel, StimRegister.EventChargeRecovOn, t_charge_recovery_on)
        self.programStimReg(
            stream, channel, StimRegister.EventChargeRecovOff, t_charge_recovery_off
        )
        self.programStimReg(stream, channel, StimRegister.EventAmpSettleOnRepeat, 0xFFFF)
        self.programStimReg(stream, channel, StimRegister.EventAmpSettleOffRepeat, 0xFFFF)
        self.programStimReg(stream, channel, StimRegister.EventEnd, t4)
        self.enableAuxCommandsOnOneStream(stream)
        reg = self.getreg(self.sampleRate)
        reg.set_dsp_cutoff_freq(0.5)
        reg.setStimStepSize(step_size)
        cmd = reg.createCommandListSetStimMagnitudes(
            magnitude_neg=int(min(255, round(amp_neg_mA * 1e6 / step_size.nA))) if enable else 0,
            magnitude_pos=int(min(255, round(amp_pos_mA * 1e6 / step_size.nA))) if enable else 0,
            channel=channel
        )
        self.uploadCommandList(cmd, 0, 0)
        self.selectAuxCommandLength(0, 0, len(cmd) - 1)

        cmd = reg.dummy(8192)
        for aux in [1, 2, 3]:
            self.uploadCommandList(cmd, aux, 0)
        self.setStimCmdMode(False)
        self.enableAuxCommandsOnOneStream(stream)
        self.runAndReadBuffer(samples=128)

        reg.set_upper_bandwidth(7500)
        reg.set_lower_bandwidth_a(1000)
        reg.set_lower_bandwidth_b(1)

        cmd = reg.createCommandListRegisterConfig(update_stim=True, readonly=True)
        self.uploadCommandList(cmd, 0, 0)
        self.selectAuxCommandLength(0, 0, len(cmd) - 1)
        self.runAndReadBuffer(samples=128)

        cmd = reg.createCommandListRegisterConfig(update_stim=True, readonly=False)
        self.uploadCommandList(cmd, 0, 0)
        self.selectAuxCommandLength(0, 0, len(cmd) - 1)
        self.enableAuxCommandsOnAllStreams()
        self.runAndReadBuffer(samples=128)

    def send_ztest_signals(
        self,
        frequency: impedance.Frequency,
        strategy: impedance.Strategy = impedance.Strategy.auto(),
        channels: List[int] = None,
        progress: bool = True,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        self.setStimCmdMode(False)
        for i in range(8):
            self.enableExternalDigOut(i, False)
        for i in range(8):
            self.enableDac(i, False)

        headstage_channels = 16 if self.rhs else 32
        test_channels = channels if channels is not None else list(range(headstage_channels))
        sample_rate = self.sampleRate.rate
        period = frequency.get_period(sample_rate)
        frequency = frequency.get_actual(sample_rate)

        reg = self.getreg(self.sampleRate)
        cmd = reg.createCommandListZcheckDac(frequency, 128, 8192)
        self.uploadCommandList(cmd, 0, 1)
        self.selectAuxCommandLength(0, 0, len(cmd) - 1)
        self.selectAuxCommandBank('all', 0, 1)
        num_periods = strategy.get_num_periods(frequency)
        numBlocks = int(np.ceil((num_periods + 2) * period / 128))
        reg.set_dsp_cutoff_freq(0.5)
        if self.rhs:
            reg.set_lower_bandwidth_b(1)
        else:
            reg.set_lower_bandwidth(1)
        reg.set_upper_bandwidth(7500)
        reg.controller.set('dspEnable', 1)
        reg.controller.set('zcheckEn', 1)
        if self.rhs:
            cmd = reg.createCommandListRegisterConfig(False, False)
        else:
            cmd = reg.createCommandListRegisterConfig(False)
        # self.uploadCommandList(cmd, 2, 3)
        self.selectAuxCommandLength(2, 0, len(cmd) - 1)
        self.selectAuxCommandBank('all', 2, 3)

        all_data = []
        for zscale in tqdm(range(3), disable=not progress, desc='scale'):
            reg.controller.set('zcheckScale', zscale)
            all_data.append([])
            for ch in tqdm(test_channels, disable=not progress, desc='channel'):
                reg.controller.set('zcheckSelect', ch)
                if self.rhs:
                    cmd = reg.createCommandListRegisterConfig(False, False)
                else:
                    cmd = reg.createCommandListRegisterConfig(False)
                self.uploadCommandList(cmd, 2, 3)
                self.runAndReadDataBlock(samples=128)  # apply the new command
                sps = self.runAndReadDataBlock(samples=numBlocks * 128).to_samples()
                if self.rhs:
                    data = sps.amp[:, :, :, 1]
                else:
                    data = sps.amp[:, :, :]
                all_data[-1].append(data)
        #         0         1       2        3       4
        #   zscale, wave pin, signal, channel, stream
        all_data = np.array(all_data).transpose(0, 1, 4, 3, 2)
        # offset by 3 commands delay
        all_data = all_data[..., 3 + 2 * period:period * num_periods + 3 - period]
        # -> zscale, wave pin, stream, channel, signal
        return all_data

    def measure_impedance(
        self,
        frequency: impedance.Frequency,
        strategy: impedance.Strategy = impedance.Strategy.auto(),
        channels: List[int] = None,
        progress: bool = True,
        raw_data_return: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Measure impedance of the headstage

        Parameters:
        -----------
        frequency: impedance.Frequency
            Target frequency to measure the impedance in Hz. The actual frequency will be
            adjusted to the nearest frequency that can be generated by the FPGA.
        strategy: impedance.Strategy
            Specify the measurement duration, see Strategy for details.
        channels: List[int]
            The channels to test, if None, test all channels.
            Note that all datastreams will be tested in parallel.
        progress: bool
            Whether to show progress bar
        raw_data_return: bool
            Whether to skip the impedance calculation and return the raw measurement data instead.

        Returns:
        --------
        When raw_data_return is False:

        magnitude: np.ndarray
            The magnitude of the impedance in Ohm, shape (n_stream, n_channel)
        phase: np.ndarray
            The phase of the impedance in degree, shape (n_stream, n_channel)
        
        When raw_data_return is True:
        raw_data: np.ndarray
        """
        headstage_channels = 16 if self.rhs else 32
        test_channels = channels if channels is not None else list(range(headstage_channels))
        all_data = self.send_ztest_signals(frequency, strategy, test_channels, progress)
        n_zscale, n_test_ch, n_stream, _, _ = all_data.shape
        assert n_zscale == 3
        # extract only signals from the target channel which the testing signal is applied
        # -> target_channel, zscale, stream, signal
        target_channel_data = all_data[:, np.arange(n_test_ch), :, test_channels, :]

        # -> zscale, stream, target_channel, signal
        target_channel_data = np.moveaxis(target_channel_data, 0, 2)

        if raw_data_return:
            return target_channel_data

        # -> zscale, stream * target_channel, signal
        signal_to_measure = target_channel_data.reshape((n_zscale, n_stream * n_test_ch, -1))
        magnitude, phase = impedance.calculate_impedance(
            signal_to_measure,
            self.getSampleRate(),
            rhs=self.rhs,
            frequency=frequency.get_actual(self.getSampleRate()),
        )

        return magnitude.reshape((n_stream, n_test_ch)), phase.reshape((n_stream, n_test_ch))


def get_XDAQ(*, rhs: bool = False, index=0, fastSettle: bool = False, skip_headstage: bool = False):
    devices = Board.list_devices()
    if index >= len(devices):
        raise IndexError(f"Device index {index} out of range (found {len(devices)} device(s))")

    xdaq = XDAQ(devices[index].with_mode('rhs' if rhs else 'rhd').create())
    xdaq.initialize()

    xdaq.changeSampleRate(SampleRate.SampleRate30000Hz, fastSettle)
    if skip_headstage:
        return xdaq

    xdaq.findConnectedAmplifiers()
    xdaq.calibrateADC(fastSettle)
    return xdaq


def getBlocksizeInWords(rhs, numDataStreams):
    if rhs:
        # 4 = magic number; 2 = time stamp; 20 = (16 amp channels + 4 aux commands, each 32 bit results);
        # 4 = stim control params; 8 = DACs; 8 = ADCs; 2 = TTL in/out
        sampleSize = 4 + 2 + numDataStreams * (2 * (16 + 4) + 4) + 8 + 8 + 2 + 2 + 2
        return sampleSize
    else:
        # 4 = magic number; 2 = time stamp; 35 = (32 amp channels + 3 aux commands); 0-3 filler words; 8 = ADCs; 4 = TTL in/out
        padding = ((numDataStreams + 2) % 4)
        sampleSize = (4 + 2 + (numDataStreams * (32 + 3)) + padding + 8 + 2 + 2)
        return sampleSize
