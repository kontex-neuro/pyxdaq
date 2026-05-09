import inspect
import math
import time
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from dataclass_wizard import JSONWizard
from pylibxdaq import pyxdaq_device
from tqdm.auto import tqdm

from . import impedance, resources
from .board import Board
from .constants import RHD, RHS, HeadstageChipID, HeadstageChipMISOID, SampleRate
from .datablock import DataBlock, Samples, get_sample_size
from .legacy import _LegacyMixin
from .rhd_driver import RHDDriver
from .rhs_driver import RHSDriver
from .stim import StimSubsystem


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
            return f'{stat} Data stream[{self.sid:02d}] - NA'
        channel_ranges = {
            HeadstageChipMISOID.MISO_A: "[ 0,31]",
            HeadstageChipMISOID.MISO_B: "[32,63]",
        }
        channel_range = channel_ranges.get(self.miso, "NA")
        return (f"{stat} Data stream[{self.sid:02d}] - "
                f"{self.chip.name}:{channel_range}")


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
                out += f'{self.portNumber:1d}- {s}\n'
            elif i == len(self.streams) - 1:
                out += f'|- {s}'
            else:
                out += f'|  {s}\n'
        return out


@dataclass
class XDAQPorts(JSONWizard):
    spi_per_port: int
    chips_per_spi: int  # MOSI
    ddr: bool
    ports: List[HdmiPort]
    streams: List[StreamConfig]
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


class XDAQ(_LegacyMixin):
    dev: Board
    ports: XDAQPorts
    rhs: bool
    ep: Union[RHD, RHS]
    stim: Union["StimSubsystem", None]
    sampleRate: SampleRate = SampleRate.SampleRate30000Hz

    def __init__(self, dev: Board):
        self.dev = dev
        self.ep = RHS if self.dev.rhs else RHD
        self.rhs = self.dev.rhs
        self.stim = StimSubsystem(self) if self.rhs else None
        self.ports = XDAQPorts.default(2, 1 if self.dev.rhs else 2, False if self.dev.rhs else True)
        if 'Device Timestamp' in self.dev.status.get('Capabilities', {}):
            self.device_timestamp = True
            # TODO: remove hardcoded register address
            self.dev.raw.set_register_sync(0x1544, 1, 1)
        else:
            self.device_timestamp = False

    def getreg(self, sample_rate: SampleRate) -> Union[RHDDriver, RHSDriver]:
        R = RHSDriver if self.rhs else RHDDriver
        res = resources.rhs if self.rhs else resources.rhd
        return R(sample_rate, res.reg_path, res.isa_path)

    def reset_board(self):
        """
        This clears all auxiliary command RAM banks, clears the USB FIFO, and resets the
        per-channel sampling rate to 30.0 kS/s/ch.
        """
        self.dev.set_register(self.ep.WireInResetRun, 1, 1)
        self.dev.set_register(self.ep.WireInResetRun, 0, 1)

    def run(self):
        self.dev.send_trigger(self.ep.TrigInSpiStart, 0)

    def resetSequencers(self):
        self.dev.send_trigger(self.ep.TrigInSpiStart, 1)

    def is_running(self):
        return self.dev.read_register(self.ep.WireOutSpiRunning) & 1

    def flush(self):
        self.dev.raw.read(0xA0, bytearray())

    def set_auxcmd_bank(self, port: Union[int, str], auxCommandSlot, bank: int):
        """
        Select an auxiliary command slot (AuxCmd1, AuxCmd2, or AuxCmd3) and bank (0-15) for a particular SPI port
        (PortA - PortH) on the FPGA.
        Bank: 4 bit
                 |0123|4567|8901|2345|6789|0123|4567|8901|
        Bank for |P0  |P1  |P2  |P3  |P4  |P5  |P6  |P7  |
        """
        if self.ep != RHD:
            return
        if auxCommandSlot < 0 or auxCommandSlot > 2:
            raise Exception("auxCommandSlot out of range")
        if bank < 0 or bank > 15:
            raise Exception("bank out of range")

        ep = [self.ep.WireInAuxCmdBank1, self.ep.WireInAuxCmdBank2,
              self.ep.WireInAuxCmdBank3][auxCommandSlot]
        if isinstance(port, str):
            if port != 'all':
                raise Exception("port must be 'all' or an integer 0-7")
            # repeat bank for all 8 ports, fill in same 4 bits bank for all ports
            bank = bank << 4 | bank
            bank = bank << 8 | bank
            bank = bank << 16 | bank
            self.dev.set_register(ep, bank)
        else:
            self.dev.set_register(ep, bank << (port * 4), 0xf << (port * 4))

    def set_auxcmd_length(self, auxCommandSlot, loopIndex, endIndex):
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
        if self.ep == RHS:
            self.dev.set_register_indirect(
                self.ep.TrigInAuxCmdLength, auxCommandSlot + 4, self.ep.WireInMultiUse, loopIndex
            )
            self.dev.set_register_indirect(
                self.ep.TrigInAuxCmdLength, auxCommandSlot, self.ep.WireInMultiUse, endIndex
            )
        else:
            self.dev.set_register(
                self.ep.WireInAuxCmdLoop, loopIndex << (auxCommandSlot * 10), 0x000003ff <<
                (auxCommandSlot * 10), False
            )
            self.dev.set_register(
                self.ep.WireInAuxCmdLength, endIndex << (auxCommandSlot * 10), 0x000003ff <<
                (auxCommandSlot * 10)
            )

    def set_sample_rate(self, sample_rate: SampleRate):
        self.sampleRate = sample_rate
        isDcmProgDone = lambda: (self.dev.read_register(self.ep.WireOutDataClkLocked) & 0x0002) > 1
        while not isDcmProgDone():
            time.sleep(0.01)
        self.dev.set_register(
            self.ep.WireInDataFreqPll, 256 * sample_rate.value[0] + sample_rate.value[1]
        )
        if self.ep == RHS:
            self.dev.send_trigger(self.ep.TrigInDcmProg, 0)
        else:
            self.dev.send_trigger(self.ep.TrigInConfig, 0)
        isDataClockLocked = lambda: (
            self.dev.read_register(self.ep.WireOutDataClkLocked) & 0x0001
        ) > 0
        while not isDataClockLocked():
            time.sleep(0.01)

    def set_continuous_run_mode(self, enable: bool):
        self.dev.set_register(self.ep.WireInResetRun, 0x02 * enable, 0x02)

    def set_max_timestep(self, maxTimeStep: int):
        if maxTimeStep < 0 or maxTimeStep > 2**32 - 1:
            raise Exception("maxTimeStep out of range")
        self.dev.set_register(self.ep.WireInMaxTimeStep, maxTimeStep)

    def set_cable_delay(self, port: Union[int, str], delay: int):
        if delay < 0 or delay > 15:
            raise Exception("delay out of range")
        if (isinstance(port, str) and port != 'all') or (isinstance(port, int) and
                                                         (port < 0 or port > 7)):
            raise Exception("port out of range")

        if isinstance(port, str) and port == 'all':
            delay = delay << 4 | delay
            delay = delay << 8 | delay
            delay = delay << 16 | delay
            self.dev.set_register(self.ep.WireInMisoDelay, delay)
        else:
            self.dev.set_register(self.ep.WireInMisoDelay, delay << (4 * port), 0xf << (4 * port))

    @staticmethod
    def delay_from_cable_length(length, sampleRate, unit):
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

    def set_dsp_settle(self, enable):
        self.dev.set_register(self.ep.WireInResetRun, enable * 0x4, 0x4)

    def config_data_stream(self, stream: Union[int, str], enable: bool, force=False):
        if isinstance(stream, str) and stream == 'all':
            self.dev.set_register(self.ep.WireInDataStreamEn, (2**32 - 1) * enable)
            # update cached value for all streams inside port object
            for s in self.ports.streams:
                s.enabled = enable
        else:
            if self.ports.streams[stream].enabled == enable:
                if not force:
                    return
            self.dev.set_register(
                self.ep.WireInDataStreamEn, (0x1 * enable) << stream, 0x1 << stream
            )
            self.ports.streams[stream].enabled = enable

    @property
    def num_enabled_datastream(self):
        # TODO: use cached value
        return sum(i.enabled for i in self.ports.streams)

    def set_ttl_override(self, enable: Union[int, bool]):
        v = enable * 0xffffffff if isinstance(enable, bool) else enable & 0xffffffff
        self.dev.set_register(self.ep.TTL_override, v)

    def set_ttl_out(self, channel: Union[int, str], enable: Union[bool, int]):
        if isinstance(channel, str) and channel == 'all':
            v = enable * 0xffffffff if isinstance(enable, bool) else enable & 0xffffffff
            self.dev.set_register(self.ep.WireInTtlOut, v)
        else:
            self.dev.set_register(self.ep.WireInTtlOut, int(enable) << channel, 1 << channel)

    def _getDacEndpoint(self, dacChannel):
        if dacChannel < 0 or dacChannel > 11:
            raise ValueError("dacChannel out of range")
        return [
            self.ep.WireInDacSource1, self.ep.WireInDacSource2, self.ep.WireInDacSource3,
            self.ep.WireInDacSource4, self.ep.WireInDacSource5, self.ep.WireInDacSource6,
            self.ep.WireInDacSource7, self.ep.WireInDacSource8, self.ep.WireInDacSource9,
            self.ep.WireInDacSource10, self.ep.WireInDacSource11, self.ep.WireInDacSource12
        ][dacChannel]

    def enable_dac(self, channel: int, enable: bool):
        if channel < 0 or channel > 11:
            raise ValueError("channel out of range")
        bm = 0x0200 if self.rhs else 0x0800
        self.dev.set_register(self._getDacEndpoint(channel), enable * bm, bm)

    def set_dac_data_stream(self, channel: bool, stream: int):
        if channel < 0 or channel > 11:
            raise ValueError("channel out of range")
        if stream < 0 or stream > (9 if self.rhs else 33):
            raise ValueError("stream out of range")
        self.dev.set_register(
            self._getDacEndpoint(channel), stream << 5, 0x1e0 if self.rhs else 0x07e0
        )

    def selectDacDataChannel(self, dacChannel: bool, dataChannel: int):
        if dacChannel < 0 or dacChannel > 7:
            raise ValueError("dacChannel out of range")
        if dataChannel < 0 or dataChannel > 31:
            raise ValueError("dataChannel out of range")
        self.dev.set_register(self._getDacEndpoint(dacChannel), dataChannel, 0x001f)

    def config_dac(self, channel: int, enable: bool, stream: int, dataChannel: int):
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
        self.dev.set_register(
            self._getDacEndpoint(channel), (enable * enablebit) | (stream << 5) | dataChannel,
            0x3fff if self.rhs else 0xffff
        )

    def set_dac_manual(self, value):
        """
        When DacSourceStream is set to 32, the DAC will be controlled directly by the host computer via WireInDacManual.
        32768 = 0V
        """
        if value < 0 or value > 65535:
            raise Exception("value out of range")
        self.dev.set_register(self.ep.WireInDacManual, value)

    def set_dac_gain(self, gain: int):
        if gain < 0 or gain > 7:
            raise Exception("gain out of range")
        self.dev.set_register(self.ep.WireInResetRun, gain << 13, 0xe000)

    def set_audio_noise_suppress(self, noiseSuppress: int):
        if noiseSuppress < 0 or noiseSuppress > 127:
            raise Exception("noiseSuppress out of range")
        self.dev.set_register(self.ep.WireInResetRun, noiseSuppress << 6, 0x1fc0)

    def set_ttl_mode(self, mode: Union[bool, List[bool]]):
        """
        Set TTL output mode for channels 0-7, True for control by FPGA.
        Only RHS can set individual TTL mode.
        """
        if self.ep == RHS:
            if isinstance(mode, bool):
                val = 0xff * mode
            elif isinstance(mode, list) and len(mode) == 8:
                val = sum(1 << i for i, v in enumerate(mode) if v)
            else:
                raise Exception(f'invalid mode: {mode}')
            self.dev.set_register(self.ep.WireInTtlOutMode, val, 0xff)
        else:
            if not isinstance(mode, bool):
                raise Exception(f'invalid mode: {mode}')
            self.dev.set_register(self.ep.WireInResetRun, 8 * mode, 8)

    def set_dac_threshold(self, channel: int, threshold: int, trigPolarity: bool):
        if channel < 0 or channel > 7:
            raise Exception("channel out of range")
        if threshold < 0 or threshold > 65535:
            raise Exception("threshold out of range")
        ep = self.ep.TrigInDacThresh if self.rhs else self.ep.TrigInDacConfig
        self.dev.set_register_indirect(ep, channel, self.ep.WireInMultiUse, threshold)
        self.dev.set_register_indirect(ep, channel + 8, self.ep.WireInMultiUse, int(trigPolarity))

    def enable_external_fast_settle(self, enable: bool):
        self.dev.set_register_indirect(self.ep.TrigInConfig, 6, self.ep.WireInMultiUse, int(enable))

    def set_external_fast_settle_channel(self, channel: int):
        if channel < 0 or channel > 15:
            raise Exception("channel out of range")
        self.dev.set_register_indirect(self.ep.TrigInConfig, 7, self.ep.WireInMultiUse, channel)

    def enable_external_dig_out(self, port: int, enable: bool):
        if self.ep != RHD:
            return
        if port < 0 or port > 7:
            raise Exception("port out of range")
        self.dev.set_register_indirect(
            self.ep.TrigInDacConfig, 16 + port, self.ep.WireInMultiUse, int(enable)
        )

    # Select which of the TTL inputs 0-15 is used to control the auxiliary digital output
    # pin of the chips connected to a particular SPI port, if external control of auxout is enabled.
    def set_external_dig_out_channel(self, port: int, channel: int):
        if self.ep != RHD:
            return
        if port < 0 or port > 7:
            raise Exception("port out of range")
        if channel < 0 or channel > 15:
            raise Exception("channel out of range")
        self.dev.set_register_indirect(
            self.ep.TrigInDacConfig, 24 + port, self.ep.WireInMultiUse, channel
        )

    def config_dac_ref(self, enable: bool, stream: int = 0, channel: int = 0):
        if stream < 0 or stream > (7 if self.rhs else 31):
            raise Exception("stream out of range")
        if channel < 0 or channel > (15 if self.rhs else 31):
            raise Exception("channel out of range")
        if self.ep == RHS:
            # this doesn't match the documentation, but RHX uses this implementation
            self.dev.set_register(
                self.ep.WireInDacReref, (enable * 0x100) | (stream << 5) | channel, 0x1fff
            )
        else:
            self.dev.set_register(
                self.ep.WireInDacReref, (enable * 0x400) | (stream << 5) | channel, 0x7fff
            )

    def enable_auxcmd_on_stream(self, stream: int | str):
        if self.ep != RHS:
            return
        if isinstance(stream, str) and stream == 'all':
            self.dev.set_register(self.ep.WireInAuxEnable, 0xff, 0xff)
        elif isinstance(stream, int) and 0 <= stream < 8:
            self.dev.set_register(self.ep.WireInAuxEnable, 1 << stream, 0xff)
        else:
            raise Exception("stream must be 'all' or an integer 0-7")

    def set_global_settle_policy(self, settle: List[bool], global_settle: bool):
        if self.ep != RHS:
            return
        if len(settle) != 4:
            raise Exception("settle must be a list of 4 booleans")
        v = sum(p * (1 << i) for i, p in enumerate(settle)) | (global_settle * 0x10)
        self.dev.set_register(self.ep.WireInGlobalSettleSelect, v, 0x1f)

    def enable_dc_amp_convert(self, enabled: bool):
        if self.ep != RHS:
            return
        self.dev.set_register(self.ep.WireInDcAmpConvert, enabled * 0x1, 0x1)

    def set_extra_states(self, states: int):
        if self.ep != RHS:
            return
        self.dev.set_register(self.ep.WireInExtraStates, states)

    def set_analog_in_trigger_threshold(self, threshold: float):
        if self.ep != RHS:
            return
        value = int(32768 * threshold / 10.24) + 32768
        value = max(0, min(65535, value))
        self.dev.set_register(self.ep.WireInAdcThreshold, value)

    def initialize(self):
        self.enable_auxcmd_on_stream('all')
        self.set_global_settle_policy([False, False, False, False], False)
        self.set_sample_rate(SampleRate.SampleRate30000Hz)
        for auxCommandSlot in range(3):
            self.set_auxcmd_bank('all', auxCommandSlot, 0)
        for auxCommandSlot in range(3 + int(self.rhs)):
            self.set_auxcmd_length(auxCommandSlot, 0, 0)
        if self.stim:
            self.stim.disable()
        self.set_continuous_run_mode(True)
        self.set_max_timestep(2**32 - 1)
        self.set_cable_delay('all', self.delay_from_cable_length(3.0, 30000, 'ft'))

        self.set_dsp_settle(False)
        self.config_data_stream('all', False, True)
        self.config_data_stream(0, True, True)

        self.enable_dc_amp_convert(True)
        self.set_extra_states(0)
        self.set_ttl_out('all', False)
        for i in range(8):
            self.config_dac(i, False, 0, 0)  # Initially point DACs to DacManual1 input
        self.set_dac_manual(32768)
        self.set_dac_gain(0)
        self.set_audio_noise_suppress(0)
        self.set_ttl_mode(False)

        for i in range(8):
            self.set_dac_threshold(i, 32768, True)
        if not self.rhs:
            self.enable_external_fast_settle(False)
            self.set_external_fast_settle_channel(0)
        for i in range(8):
            self.enable_external_dig_out(i, False)
        for i in range(8):
            self.set_external_dig_out_channel(i, 0)
        self.config_dac_ref(False)
        self.enable_dac_highpass_filter(False)

        self.set_analog_in_trigger_threshold(1.65)
        if self.stim:
            self.stim.reset_sequencers()

    def upload_auxcmd(self, commandList: np.ndarray, auxCommandSlot, bank):
        if auxCommandSlot < 0 or auxCommandSlot > (2 + int(self.rhs)):
            raise Exception("auxCommandSlot out of range")
        if bank < 0 or bank > 15:
            raise Exception("bank out of range")
        if self.ep == RHS:
            self.dev.send_trigger(self.ep.TrigInRamAddrReset, 0)
            ep = [
                self.ep.PipeInAuxCmd1, self.ep.PipeInAuxCmd2, self.ep.PipeInAuxCmd3,
                self.ep.PipeInAuxCmd4
            ][auxCommandSlot]
            commandList = np.pad(commandList, (0, 16 - len(commandList) % 16), 'constant')
            self.dev.write_data(ep, bytearray(commandList.tobytes(order='C')))
        else:
            self.dev.set_register(self.ep.WireInCmdRamBank, bank)
            for i, cmd in enumerate(commandList):
                self.dev.set_register(self.ep.WireInCmdRamData, int(cmd), update=False)
                self.dev.set_register(self.ep.WireInCmdRamAddr, i)
                self.dev.send_trigger(self.ep.TrigInConfig, auxCommandSlot + 1)

    def set_dac_highpass_filter(self, cutoff: float, smapleRate: float):
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
        self.dev.set_register_indirect(
            self.ep.TrigInConfig, 5, self.ep.WireInMultiUse, filterCoefficient
        )

    def upload_commands(
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
        if isinstance(reg, RHDDriver):
            reg.upload_commands(self, fastSettle, upper_bandwidth, lower_bandwidth)
        elif isinstance(reg, RHSDriver):
            reg.upload_commands(
                self,
                update_stim=stim_params['update_stim'],
                readonly=stim_params['readonly'],
                upper_bandwidth=upper_bandwidth,
                lower_bandwidth=lower_bandwidth,
            )

    def update_sample_rate(
        self,
        sampleRate: SampleRate,
        fastSettle: bool = False,
        update_stim: bool = False,
        upper_bandwidth: float = 7500,
        lower_bandwidth: float = 1
    ):
        self.set_sample_rate(sampleRate)
        if not self.rhs:
            self.set_dac_highpass_filter(250, sampleRate.rate)

        self.upload_commands(
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

    def _get_dac_ep(self, dacChannel: int):
        if dacChannel < 0 or dacChannel > 11:
            raise RuntimeError("dacChannel out of range")
        return [
            self.ep.PipeInDAC1, self.ep.PipeInDAC2, self.ep.PipeInDAC3, self.ep.PipeInDAC4,
            self.ep.PipeInDAC5, self.ep.PipeInDAC6, self.ep.PipeInDAC7, self.ep.PipeInDAC8,
            self.ep.PipeInDAC9, self.ep.PipeInDAC10, self.ep.PipeInDAC11, self.ep.PipeInDAC12
        ][dacChannel]

    def upload_dac_data(
        self, waveform: np.ndarray, dacChannel: int, length: int, from_voltage: bool = True
    ):
        buffer = bytearray(self._encodeWaveform(waveform, from_voltage))
        self.dev.send_trigger(self.ep.TrigInSpiStart, 2)
        result = self.dev.write_data(self._get_dac_ep(dacChannel), buffer)
        if result < 0:
            raise RuntimeError("Upload waveform failed")
        if dacChannel <= 7:
            self.dev.set_register_indirect(
                self.ep.TrigInSpiStart, 8 + dacChannel, self.ep.WireInMultiUse, length
            )

    def upload_dac_waveform(self, waveform, channel: int, length: int, from_voltage: bool = True):

        self.upload_dac_data(waveform, channel, length, from_voltage)
        self.set_dac_data_stream(channel, 9 if self.rhs else 33)
        self.enable_dac(channel, True)

    def start(self, *, continuous: bool = None):
        if self.stim:
            self.stim.enable()
        if continuous is not None:
            self.set_continuous_run_mode(continuous)
        self.run()

    def stop(self, *, wait: bool = False):
        if self.stim:
            self.stim.disable()
        self.set_max_timestep(0)
        self.set_continuous_run_mode(False)
        if wait:
            start = time.time()
            while self.is_running():
                time.sleep(0.01)
                if time.time() - start > 1:
                    raise TimeoutError("Timeout waiting for stop")

    def acquire_raw_data(self, samples) -> bytes:
        self.flush()
        buffers = []

        sample_size = self.sample_size_in_bytes()
        total_size = samples * sample_size
        error = None

        def callback(data: pyxdaq_device.DataView | None, e: str | None):
            nonlocal error
            try:
                if e is not None:
                    error = e
                if data is not None:
                    buffers.append(data.numpy)
            except Exception as e:
                error = str(e)

        self.set_max_timestep(0)
        self.set_continuous_run_mode(True)
        sample_time = samples / self.sample_rate_hz
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
            while (sum(map(len, buffers)) < total_size) and (error is None):
                if time.time() - start > (1 + sample_time):
                    raise TimeoutError("Timeout getting samples")
                time.sleep(0.01)
            self.set_continuous_run_mode(False)

        if error is not None:
            raise RuntimeError(f"Error receiving data: {error}")

        self.flush()

        return np.concatenate(buffers)[:total_size]

    def acquire_samples(self, samples) -> Samples:
        buffer = self.acquire_raw_data(samples)
        return DataBlock.from_buffer(
            self.rhs, self.sample_size_in_bytes(), buffer, self.num_enabled_datastream,
            self.device_timestamp
        ).to_samples()

    def start_receiving_buffer(
        self,
        callback: Callable[[pyxdaq_device.DataView | None, str | None], None],
    ):
        sample_size = self.sample_size_in_bytes()
        sample_rate = self.sample_rate_hz

        hardware_events_per_sec = 100
        chunk_size = int(sample_size * sample_rate / hardware_events_per_sec)

        return self.dev.start_receiving_aligned_buffer(
            self.ep.PipeOutData, sample_size, callback, chunk_size=chunk_size
        )

    def start_receiving_samples(
        self,
        callbacks: List[Union[
            Callable[[Samples], None],
            Callable[[Samples | None, str | None], None],
        ]],
        on_error: Optional[Callable[[str], None]],
    ):
        """
        Starts receiving data and provides parsed Samples objects to callbacks.

        This is a high-level interface that handles byte buffer parsing internally.
        If an exception occurs within a user-provided callback, it is caught and
        reported via the `on_error` function. This prevents a single faulty
        callback from halting the entire data processing pipeline; other
        callbacks will continue to be executed for subsequent data chunks.

        Args:
            callbacks: A list of callback functions to be invoked with data.
                Each callback can have one of two signatures:
                - my_callback(samples: Samples): Called on successful data receipt.
                - my_callback(samples: Samples | None, error: str | None): Called
                  for both data and errors.
            on_error: An optional callback that is invoked only when an error occurs.
        """
        if not callbacks:
            raise ValueError("At least one callback must be provided")

        sample_callbacks = []
        sample_error_callbacks = []

        for i, cb in enumerate(callbacks):
            if not callable(cb):
                raise ValueError(f"Callback at index {i} is not callable")

            arity = len(inspect.signature(cb).parameters)

            if arity == 1:
                sample_callbacks.append(cb)
            elif arity == 2:
                sample_error_callbacks.append(cb)
            else:
                cb_name = getattr(cb, '__name__', f'<callback_{i}>')
                raise ValueError(f"Callback {cb_name} has invalid signature")

        def _internal_callback(data: pyxdaq_device.DataView | None, error: str | None):
            try:
                if error is not None:
                    on_error(error)
                    for cb in sample_error_callbacks:
                        cb(None, error)
                    return

                if data is None:
                    return

                try:
                    samples = DataBlock.from_buffer(
                        self.rhs, self.sample_size_in_bytes(), data.numpy,
                        self.num_enabled_datastream, self.device_timestamp
                    ).to_samples()
                    for cb in sample_callbacks:
                        cb(samples)
                    for cb in sample_error_callbacks:
                        cb(samples, None)
                except Exception as e:
                    parse_error = f"Failed to parse data block: {e}"
                    on_error(parse_error)
                    for cb in sample_error_callbacks:
                        cb(None, parse_error)

            except Exception as e:
                on_error(f"Unhandled exception in callback: {e}")

        return self.start_receiving_buffer(_internal_callback)

    def test_cable_delay(self, output: str = ''):
        headstagename = np.array([ord(i) for i in ('INTAN' if self.rhs else 'INTANRHD')])
        headstageids = np.array([i.value for i in HeadstageChipID])
        n_streams = 8 if self.rhs else 16
        if self.rhs:
            for stream in range(8):
                self.config_data_stream(stream, True)
        else:
            for stream in range(32):
                self.config_data_stream(stream, (stream % 2) == 0)
        self.set_auxcmd_bank('all', 2, 0)
        results = []
        for delay in range(16):
            self.set_cable_delay('all', delay)
            sp = self.acquire_samples(128)
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
        self.set_auxcmd_bank('all', 2, 1)
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

    def scan_headstages(self):
        self.ports = XDAQPorts.fromChipInfos(
            self.test_cable_delay(), 2, 1 if self.rhs else 2, not self.rhs
        )
        for spi, streams in enumerate(
                self.ports.group_by_port() if self.rhs else self.ports.group_by_spi()):
            self.set_cable_delay(spi, max(s.delay for s in streams))
        # since ports are replaced, we need a force update to sync the enable state
        # it could be done by copying the cached state from the old ports
        self.config_data_stream('all', False, True)
        for s in self.ports.streams:
            self.config_data_stream(s.sid, s.available)
        self.set_spi_led_indicator([any(c.available for c in p) for p in self.ports])

    def set_spi_led_indicator(self, stat: List[bool]):
        value = sum(1 << i for i, v in enumerate(stat) if v)
        self.dev.set_register_indirect(self.ep.TrigInConfig, 8, self.ep.WireInMultiUse, value)

    def enable_dac_highpass_filter(self, enable: bool):
        self.dev.set_register_indirect(
            self.ep.TrigInConfig, 4, self.ep.WireInMultiUse, 1 if enable else 0
        )

    def sample_size_in_bytes(self):
        return get_sample_size(self.rhs, self.num_enabled_datastream, self.device_timestamp)

    @property
    def sample_rate_hz(self) -> int | float:
        return self.sampleRate.value[2]

    def calibrate_adc(self, fastSettle: bool = False):
        # RHD: Select RAM Bank 0 for AuxCmd3 initially, so the ADC is calibrated.
        # RHS: use CLEAR command to calibrate ADC
        self.set_auxcmd_bank('all', 2, 0)
        self.acquire_raw_data(samples=128)  # upload aux commands
        self.set_auxcmd_bank('all', 2, 2 if fastSettle else 1)

    def send_ztest_signals(
        self,
        frequency: impedance.Frequency,
        strategy: impedance.Strategy = impedance.Strategy.auto(),
        channels: Optional[List[int]] = None,
        progress: bool = True,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        if self.stim:
            self.stim.disable()
        for i in range(8):
            self.enable_external_dig_out(i, False)
        for i in range(8):
            self.enable_dac(i, False)

        headstage_channels = 16 if self.rhs else 32
        test_channels = channels if channels is not None else list(range(headstage_channels))
        sample_rate = self.sampleRate.rate
        period = frequency.get_period(sample_rate)
        frequency = frequency.get_actual(sample_rate)

        reg = self.getreg(self.sampleRate)
        cmd = reg.createCommandListZcheckDac(frequency, 128, 8192)
        self.upload_auxcmd(cmd, 0, 1)
        self.set_auxcmd_length(0, 0, len(cmd) - 1)
        self.set_auxcmd_bank('all', 0, 1)
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
        # self.upload_auxcmd(cmd, 2, 3)
        self.set_auxcmd_length(2, 0, len(cmd) - 1)
        self.set_auxcmd_bank('all', 2, 3)

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
                self.upload_auxcmd(cmd, 2, 3)
                self.acquire_raw_data(samples=128)  # apply the new command
                sps = self.acquire_samples(samples=numBlocks * 128)
                if self.rhs:
                    data = sps.amp[:, :, :, 1]
                else:
                    data = sps.amp[:, :, :]
                all_data[-1].append(data)
        #         0         1       2        3       4
        #   zscale, wave pin, signal, stream, channel
        all_data = np.array(all_data).transpose(0, 1, 3, 4, 2)
        # offset by 3 commands delay
        all_data = all_data[..., 3 + 2 * period:period * num_periods + 3 - period]
        # -> zscale, wave pin, stream, channel, signal
        return all_data

    def measure_impedance(
        self,
        frequency: impedance.Frequency,
        strategy: impedance.Strategy = impedance.Strategy.auto(),
        channels: Optional[List[int]] = None,
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
            self.sample_rate_hz,
            rhs=self.rhs,
            frequency=frequency.get_actual(self.sample_rate_hz),
        )

        return magnitude.reshape((n_stream, n_test_ch)), phase.reshape((n_stream, n_test_ch))


def get_XDAQ(*, rhs: bool = False, index=0, fastSettle: bool = False, skip_headstage: bool = False):
    devices = Board.list_devices()
    if index >= len(devices):
        raise IndexError(f"Device index {index} out of range (found {len(devices)} device(s))")

    xdaq = XDAQ(devices[index].with_mode('rhs' if rhs else 'rhd').create())
    xdaq.initialize()

    xdaq.update_sample_rate(SampleRate.SampleRate30000Hz, fastSettle)
    if skip_headstage:
        return xdaq

    xdaq.scan_headstages()
    xdaq.calibrate_adc(fastSettle)
    return xdaq
