"""
Legacy API compatibility mixin for pyxdaq < 0.6.0.

Defines ``_LegacyMixin`` which is mixed into ``XDAQ`` directly, so the
deprecated camelCase methods are available on every ``XDAQ`` instance.
Each deprecated method emits a ``DeprecationWarning`` and delegates to
the current snake_case implementation.

Will be removed in a future major release.
"""

import warnings
from typing import TYPE_CHECKING

from .datablock import DataBlock

if TYPE_CHECKING:
    from .xdaq import XDAQ


def _deprecated(old: str, new: str):
    """Emit a DeprecationWarning pointing at the caller's frame."""
    warnings.warn(
        f"{old}() is deprecated and will be removed in a future release. "
        f"Use {new}() instead.",
        DeprecationWarning,
        stacklevel=3,
    )


class _LegacyMixin:
    """Mixin that adds pre-0.6 camelCase aliases onto XDAQ."""

    # ── Headstage / init ──────────────────────────────────────────────────

    def find_connected_headstages(self: "XDAQ"):
        _deprecated("find_connected_headstages", "scan_headstages")
        return self.scan_headstages()

    def calibrateADC(self: "XDAQ", fastSettle: bool = False):
        _deprecated("calibrateADC", "calibrate_adc")
        return self.calibrate_adc(fastSettle)

    # ── Sample rate ───────────────────────────────────────────────────────

    def changeSampleRate(self: "XDAQ", sampleRate, fastSettle=False, **kwargs):
        _deprecated("changeSampleRate", "update_sample_rate")
        return self.update_sample_rate(sampleRate, fastSettle, **kwargs)

    def setSampleRate(self: "XDAQ", sample_rate):
        _deprecated("setSampleRate", "set_sample_rate")
        return self.set_sample_rate(sample_rate)

    def getSampleRate(self: "XDAQ"):
        _deprecated("getSampleRate", "sample_rate_hz (property)")
        return self.sample_rate_hz

    def getSampleSizeBytes(self: "XDAQ"):
        _deprecated("getSampleSizeBytes", "sample_size_in_bytes")
        return self.sample_size_in_bytes()

    # ── Data acquisition ──────────────────────────────────────────────────

    def runAndReadDataBlock(self: "XDAQ", samples) -> DataBlock:
        _deprecated("runAndReadDataBlock", "acquire_samples")
        return DataBlock.from_buffer(
            self.rhs,
            self.sample_size_in_bytes(),
            self.acquire_raw_data(samples),
            self.num_enabled_datastream,
            self.device_timestamp,
        )

    def runAndReadBuffer(self: "XDAQ", samples) -> bytes:
        _deprecated("runAndReadBuffer", "acquire_raw_data")
        return self.acquire_raw_data(samples)

    def readDataToBuffer(self: "XDAQ", buffer: bytearray):
        _deprecated("readDataToBuffer", "acquire_raw_data")
        return self.dev.read_data(self.ep.PipeOutData, 1024, buffer)

    # ── Stim ──────────────────────────────────────────────────────────────

    def setStimCmdMode(self: "XDAQ", enabled: bool):
        _deprecated("setStimCmdMode", "set_stim_enable")
        return self.set_stim_enable(enabled)

    def programStimReg(self: "XDAQ", stream, channel, reg, value):
        _deprecated("programStimReg", "program_stim_reg")
        return self.program_stim_reg(stream, channel, reg, value)

    # ── Data streams ──────────────────────────────────────────────────────

    def enableDataStream(self: "XDAQ", stream, enable: bool, force=False):
        _deprecated("enableDataStream", "config_data_stream")
        return self.config_data_stream(stream, enable, force)

    @property
    def numDataStream(self: "XDAQ"):
        _deprecated("numDataStream", "num_enabled_datastream")
        return self.num_enabled_datastream

    # ── TTL ───────────────────────────────────────────────────────────────

    def setTTLout(self: "XDAQ", channel, enable):
        _deprecated("setTTLout", "set_ttl_out")
        return self.set_ttl_out(channel, enable)

    def setTTLMode(self: "XDAQ", mode):
        _deprecated("setTTLMode", "set_ttl_mode")
        return self.set_ttl_mode(mode)

    # ── DAC ───────────────────────────────────────────────────────────────

    def enableDac(self: "XDAQ", channel: int, enable: bool):
        _deprecated("enableDac", "enable_dac")
        return self.enable_dac(channel, enable)

    def selectDacDataStream(self: "XDAQ", channel, stream: int):
        _deprecated("selectDacDataStream", "set_dac_data_stream")
        return self.set_dac_data_stream(channel, stream)

    def selectDacDataChannel(self: "XDAQ", dacChannel, dataChannel: int):
        _deprecated("selectDacDataChannel", "config_dac")
        # no direct 1:1 mapping; forward to the underlying register write
        ep = self._getDacEndpoint(dacChannel) if hasattr(self, '_getDacEndpoint'
                                                        ) else self._get_dac_ep(dacChannel)
        self.dev.set_register(ep, dataChannel & 0x1f, 0x1f)

    def configDac(self: "XDAQ", channel, enable, stream, dataChannel):
        _deprecated("configDac", "config_dac")
        return self.config_dac(channel, enable, stream, dataChannel)

    def setDacManual(self: "XDAQ", value):
        _deprecated("setDacManual", "set_dac_manual")
        return self.set_dac_manual(value)

    def setDacGain(self: "XDAQ", gain: int):
        _deprecated("setDacGain", "set_dac_gain")
        return self.set_dac_gain(gain)

    def setDacThreshold(self: "XDAQ", channel, threshold, trigPolarity):
        _deprecated("setDacThreshold", "set_dac_threshold")
        return self.set_dac_threshold(channel, threshold, trigPolarity)

    def setDacHighpassFilter(self: "XDAQ", cutoff: float, sampleRate: float):
        _deprecated("setDacHighpassFilter", "set_dac_highpass_filter")
        return self.set_dac_highpass_filter(cutoff, sampleRate)

    def enableDacHighpassFilter(self: "XDAQ", enable: bool):
        _deprecated("enableDacHighpassFilter", "enable_dac_highpass_filter")
        return self.enable_dac_highpass_filter(enable)

    def uploadDACData(self: "XDAQ", waveform, dacChannel, length, from_voltage=True):
        _deprecated("uploadDACData", "upload_dac_data")
        return self.upload_dac_data(waveform, dacChannel, length, from_voltage)

    def uploadWaveform(self: "XDAQ", waveform, channel, length, from_voltage=True):
        _deprecated("uploadWaveform", "upload_dac_waveform")
        return self.upload_dac_waveform(waveform, channel, length, from_voltage)

    # ── Aux commands ─────────────────────────────────────────────────────

    def uploadCommandList(self: "XDAQ", commandList, auxCommandSlot, bank):
        _deprecated("uploadCommandList", "upload_auxcmd")
        return self.upload_auxcmd(commandList, auxCommandSlot, bank)

    def uploadCommands(self: "XDAQ", fastSettle=False, **kwargs):
        _deprecated("uploadCommands", "upload_commands")
        return self.upload_commands(fastSettle, **kwargs)

    def selectAuxCommandBank(self: "XDAQ", port, auxCommandSlot, bank):
        _deprecated("selectAuxCommandBank", "set_auxcmd_bank")
        return self.set_auxcmd_bank(port, auxCommandSlot, bank)

    def selectAuxCommandLength(self: "XDAQ", auxCommandSlot, loopIndex, endIndex):
        _deprecated("selectAuxCommandLength", "set_auxcmd_length")
        return self.set_auxcmd_length(auxCommandSlot, loopIndex, endIndex)

    def enableAuxCommandsOnAllStreams(self: "XDAQ"):
        _deprecated("enableAuxCommandsOnAllStreams", "enable_auxcmd_on_stream('all')")
        return self.enable_auxcmd_on_stream('all')

    def enableAuxCommandsOnOneStream(self: "XDAQ", stream):
        _deprecated("enableAuxCommandsOnOneStream", "enable_auxcmd_on_stream(stream)")
        return self.enable_auxcmd_on_stream(stream)

    # ── Cable delay ───────────────────────────────────────────────────────

    def setCableDelay(self: "XDAQ", port, delay):
        _deprecated("setCableDelay", "set_cable_delay")
        return self.set_cable_delay(port, delay)

    @staticmethod
    def delayFromCableLength(length, sampleRate, unit):
        _deprecated("delayFromCableLength", "delay_from_cable_length")
        return XDAQ.delay_from_cable_length(length, sampleRate, unit)

    def testCableDelay(self: "XDAQ", output=''):
        _deprecated("testCableDelay", "test_cable_delay")
        return self.test_cable_delay(output)

    # ── Misc ─────────────────────────────────────────────────────────────

    def setContinuousRunMode(self: "XDAQ", enable: bool):
        _deprecated("setContinuousRunMode", "set_continuous_run_mode")
        return self.set_continuous_run_mode(enable)

    def setMaxTimeStep(self: "XDAQ", maxTimeStep: int):
        _deprecated("setMaxTimeStep", "set_max_timestep")
        return self.set_max_timestep(maxTimeStep)

    def setDspSettle(self: "XDAQ", enable):
        _deprecated("setDspSettle", "set_dsp_settle")
        return self.set_dsp_settle(enable)

    def setAudioNoiseSuppress(self: "XDAQ", noiseSuppress: int):
        _deprecated("setAudioNoiseSuppress", "set_audio_noise_suppress")
        return self.set_audio_noise_suppress(noiseSuppress)

    def setGlobalSettlePolicy(self: "XDAQ", settle, global_settle):
        _deprecated("setGlobalSettlePolicy", "set_global_settle_policy")
        return self.set_global_settle_policy(settle, global_settle)

    def enableDcAmpConvert(self: "XDAQ", enabled: bool):
        _deprecated("enableDcAmpConvert", "enable_dc_amp_convert")
        return self.enable_dc_amp_convert(enabled)

    def setExtraStates(self: "XDAQ", states: int):
        _deprecated("setExtraStates", "set_extra_states")
        return self.set_extra_states(states)

    def setAnalogInTriggerThreshold(self: "XDAQ", threshold: float):
        _deprecated("setAnalogInTriggerThreshold", "set_analog_in_trigger_threshold")
        return self.set_analog_in_trigger_threshold(threshold)

    def enableExternalFastSettle(self: "XDAQ", enable: bool):
        _deprecated("enableExternalFastSettle", "enable_external_fast_settle")
        return self.enable_external_fast_settle(enable)

    def setExternalFastSettleChannel(self: "XDAQ", channel: int):
        _deprecated("setExternalFastSettleChannel", "set_external_fast_settle_channel")
        return self.set_external_fast_settle_channel(channel)

    def enableExternalDigOut(self: "XDAQ", port: int, enable: bool):
        _deprecated("enableExternalDigOut", "enable_external_dig_out")
        return self.enable_external_dig_out(port, enable)

    def setExternalDigOutChannel(self: "XDAQ", port: int, channel: int):
        _deprecated("setExternalDigOutChannel", "set_external_dig_out_channel")
        return self.set_external_dig_out_channel(port, channel)

    def setSpiLedDisplay(self: "XDAQ", stat):
        _deprecated("setSpiLedDisplay", "set_spi_led_indicator")
        return self.set_spi_led_indicator(stat)
