from __future__ import annotations

from typing import TYPE_CHECKING

from .constants import (
    RHS, StartPolarity, StimRegister, StimShape, StimStepSize, TriggerEvent, TriggerPolarity
)

if TYPE_CHECKING:
    from .xdaq import XDAQ


def _stim_trigger(source: int, event: TriggerEvent, polarity: TriggerPolarity, enabled: bool):
    return source | event.value << 5 | polarity.value << 6 | enabled << 7


def _stim_params(pulses: int, shape: StimShape, start_polarity: StartPolarity):
    return (pulses - 1) | shape.value << 8 | start_polarity.value << 10


class StimSubsystem:
    """
    Stimulation subsystem for RHS-based XDAQ devices.

    Manages the stimulation sequencer, manual triggers, and per-channel
    stimulation configuration.
    """

    def __init__(self, xdaq: "XDAQ"):
        self._xdaq = xdaq

    def enable(self):
        """Enable stimulation command mode on the FPGA."""
        self._xdaq.dev.set_register(RHS.WireInStimCmdMode, 0x1, 0x1)

    def disable(self):
        """Disable stimulation command mode on the FPGA."""
        self._xdaq.dev.set_register(RHS.WireInStimCmdMode, 0x0, 0x1)

    def trigger(self, trigger: int, on: bool):
        """
        Set manual trigger bit for a trigger channel (0-7).

        Args:
            trigger: Trigger channel index (0-7).
            on: True to assert, False to de-assert.
        """
        if trigger < 0 or trigger > 7:
            raise ValueError("trigger channel must be 0-7")
        self._xdaq.dev.set_register(RHS.WireInManualTriggers, int(on) << trigger, 1 << trigger)

    def program_reg(self, stream: int, channel: int, reg: StimRegister, value: int):
        """
        Program a stimulation register for a given stream and channel.

        WireInStimRegAddr[3:0]: StimRegAddress
        WireInStimRegAddr[7:4]: StimRegChannel 0-15
        WireInStimRegAddr[12:8]: StimRegModule 0-7 spi
        WireInStimRegWord[15:0]: StimProgWord
        """
        self._xdaq.dev.set_register(
            RHS.WireInStimRegAddr, (stream << 8) | (channel << 4) | reg.value, update=False
        )
        self._xdaq.dev.set_register(RHS.WireInStimRegWord, value)
        self._xdaq.dev.send_trigger(RHS.TrigInRamAddrReset, 1)

    def reset_sequencers(self):
        """
        Initialize all stimulation sequencers to safe defaults.

        Programs all headstage channels (streams 0-7, channels 0-15) with
        disabled triggers and NEVER-firing events, then sets up DAC streams
        (8-16) with default monophasic/biphasic patterns.
        """
        NEVER = 0xFFFF
        for stream in range(8):
            for channel in range(16):
                self.program_reg(
                    stream, channel, StimRegister.Trigger,
                    _stim_trigger(0, TriggerEvent.Edge, TriggerPolarity.Low, False)
                )
                self.program_reg(
                    stream, channel, StimRegister.Param,
                    _stim_params(1, StimShape.Biphasic, StartPolarity.cathodic)
                )
                self.program_reg(stream, channel, StimRegister.EventAmpSettleOn, NEVER)
                self.program_reg(stream, channel, StimRegister.EventStartStim, NEVER)
                self.program_reg(stream, channel, StimRegister.EventStimPhase2, NEVER)
                self.program_reg(stream, channel, StimRegister.EventStimPhase3, NEVER)
                self.program_reg(stream, channel, StimRegister.EventEndStim, NEVER)
                self.program_reg(stream, channel, StimRegister.EventRepeatStim, NEVER)
                self.program_reg(stream, channel, StimRegister.EventAmpSettleOff, NEVER)
                self.program_reg(stream, channel, StimRegister.EventChargeRecovOn, NEVER)
                self.program_reg(stream, channel, StimRegister.EventChargeRecovOff, NEVER)
                self.program_reg(stream, channel, StimRegister.EventAmpSettleOnRepeat, NEVER)
                self.program_reg(stream, channel, StimRegister.EventAmpSettleOffRepeat, NEVER)
                self.program_reg(stream, channel, StimRegister.EventEnd, 65534)

        for stream in range(8, 16):
            self.program_reg(
                stream, 0, StimRegister.Trigger,
                _stim_trigger(0, TriggerEvent.Edge, TriggerPolarity.Low, False)
            )
            self.program_reg(
                stream, 0, StimRegister.Param,
                _stim_params(1, StimShape.Monophasic, StartPolarity.cathodic)
            )
            self.program_reg(stream, 0, StimRegister.EventStartStim, 0)
            self.program_reg(stream, 0, StimRegister.EventStimPhase2, NEVER)
            self.program_reg(stream, 0, StimRegister.EventStimPhase3, NEVER)
            self.program_reg(stream, 0, StimRegister.EventEndStim, 200)
            self.program_reg(stream, 0, StimRegister.EventRepeatStim, NEVER)
            self.program_reg(stream, 0, StimRegister.EventEnd, 240)
            self.program_reg(stream, 0, StimRegister.EventChargeRecovOn, 32768)
            self.program_reg(stream, 0, StimRegister.EventChargeRecovOff, 32768 + 3200)
            self.program_reg(stream, 0, StimRegister.EventAmpSettleOnRepeat, 32768 - 3200)

        for channel in range(16):
            self.program_reg(
                16, channel, StimRegister.Trigger,
                _stim_trigger(0, TriggerEvent.Edge, TriggerPolarity.Low, False)
            )
            self.program_reg(
                16, channel, StimRegister.Param,
                _stim_params(3, StimShape.Biphasic, StartPolarity.cathodic)
            )
            self.program_reg(16, channel, StimRegister.EventStartStim, NEVER)
            self.program_reg(16, channel, StimRegister.EventEndStim, NEVER)
            self.program_reg(16, channel, StimRegister.EventRepeatStim, NEVER)
            self.program_reg(16, channel, StimRegister.EventEnd, 65534)

    def configure(
        self, stream: int, channel: int, polarity: StartPolarity, shape: StimShape, delay_ms: float,
        duration_phase1_ms: float, duration_phase2_ms: float, duration_phase3_ms: float,
        amp_neg_mA: float, amp_pos_mA: float, pulses: int, duration_pulse_ms: float,
        pre_ampsettle_ms: float, post_ampsettle_ms: float, trigger: TriggerEvent,
        trigger_source: int, trigger_pol: TriggerPolarity, step_size: StimStepSize, enable: bool,
        post_charge_recovery_ms: float
    ):
        """
        Configure stimulation parameters for a channel and upload to hardware.

        This programs the stim sequencer registers, configures current magnitudes
        via aux commands, and applies the configuration with a short acquisition.
        """
        xdaq = self._xdaq
        dt = 1 / xdaq.sampleRate.value[2]
        if not enable:
            amp_neg_mA = 0
            amp_pos_mA = 0
        self.program_reg(
            stream, channel, StimRegister.Trigger,
            _stim_trigger(trigger_source, trigger, trigger_pol, enable)
        )
        self.program_reg(stream, channel, StimRegister.Param, _stim_params(pulses, shape, polarity))
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

        self.program_reg(stream, channel, StimRegister.EventAmpSettleOn, t_ampsettle_on)
        self.program_reg(stream, channel, StimRegister.EventStartStim, t0)
        self.program_reg(stream, channel, StimRegister.EventStimPhase2, t1)
        self.program_reg(stream, channel, StimRegister.EventStimPhase3, t2)
        self.program_reg(stream, channel, StimRegister.EventEndStim, t3)
        self.program_reg(stream, channel, StimRegister.EventRepeatStim, t4)
        self.program_reg(stream, channel, StimRegister.EventAmpSettleOff, t_ampsettle_off)
        self.program_reg(stream, channel, StimRegister.EventChargeRecovOn, t_charge_recovery_on)
        self.program_reg(stream, channel, StimRegister.EventChargeRecovOff, t_charge_recovery_off)
        self.program_reg(stream, channel, StimRegister.EventAmpSettleOnRepeat, 0xFFFF)
        self.program_reg(stream, channel, StimRegister.EventAmpSettleOffRepeat, 0xFFFF)
        self.program_reg(stream, channel, StimRegister.EventEnd, t4)

        xdaq.enable_auxcmd_on_stream(stream)
        reg = xdaq.getreg(xdaq.sampleRate)
        reg.set_dsp_cutoff_freq(0.5)
        reg.setStimStepSize(step_size)
        cmd = reg.createCommandListSetStimMagnitudes(
            magnitude_neg=int(min(255, round(amp_neg_mA * 1e6 / step_size.nA))) if enable else 0,
            magnitude_pos=int(min(255, round(amp_pos_mA * 1e6 / step_size.nA))) if enable else 0,
            channel=channel
        )
        xdaq.upload_auxcmd(cmd, 0, 0)
        xdaq.set_auxcmd_length(0, 0, len(cmd) - 1)

        cmd = reg.dummy(8192)
        for aux in [1, 2, 3]:
            xdaq.upload_auxcmd(cmd, aux, 0)
        self.disable()
        xdaq.enable_auxcmd_on_stream(stream)
        xdaq.acquire_raw_data(samples=128)

        reg.set_upper_bandwidth(7500)
        reg.set_lower_bandwidth_a(1000)
        reg.set_lower_bandwidth_b(1)

        cmd = reg.createCommandListRegisterConfig(update_stim=True, readonly=True)
        xdaq.upload_auxcmd(cmd, 0, 0)
        xdaq.set_auxcmd_length(0, 0, len(cmd) - 1)
        xdaq.acquire_raw_data(samples=128)

        cmd = reg.createCommandListRegisterConfig(update_stim=True, readonly=False)
        xdaq.upload_auxcmd(cmd, 0, 0)
        xdaq.set_auxcmd_length(0, 0, len(cmd) - 1)
        xdaq.enable_auxcmd_on_stream('all')
        xdaq.acquire_raw_data(samples=128)


def enable_stim(
    xdaq: "XDAQ",
    *,
    # channel settings
    stream: int,
    channel: int,
    # current settings
    step_size: StimStepSize,
    amp_neg_mA: float,
    amp_pos_mA: float,
    # trigger settings
    trigger: TriggerEvent,
    trigger_source: int,
    trigger_pol: TriggerPolarity,
    pulses: int,
    # shape settings
    polarity: StartPolarity,
    shape: StimShape,
    # timing settings
    pre_ampsettle_ms: float,
    delay_ms: float,
    phase1_ms: float,
    phase2_ms: float,
    phase3_ms: float,
    post_pulse_ms: float,
    post_ampsettle_ms: float,
    post_charge_recovery_ms: float,
):
    max_current = step_size.nA * 256
    assert 0 <= amp_neg_mA and 0 <= amp_pos_mA, 'current must be positive'

    if amp_neg_mA * 1e6 > max_current:
        raise Exception(
            f'negative current out of range, max {step_size.nA} * 256 = {max_current} nA'
        )

    if 0 < amp_neg_mA * 1e6 < step_size.nA:
        print(f'WARNING: negative current is less than one step size ({step_size.nA} nA)')

    if amp_pos_mA * 1e6 > max_current:
        raise Exception(
            f'positive current out of range, max {step_size.nA} * 256 = {max_current} nA'
        )
    if 0 < amp_pos_mA * 1e6 < step_size.nA:
        print(f'WARNING: positive current is less than one step size ({step_size.nA} nA)')

    if shape == StimShape.Biphasic and phase2_ms == 0 or shape == StimShape.BiphasicWithInterphaseDelay:
        raise Exception('Biphasic shape requires duration_phase2_ms > 0')
    elif shape == StimShape.Monophasic and phase2_ms != 0:
        raise Exception('duration_phase2_ms > 0 only allowed for Biphasic shape')

    if shape == StimShape.Triphasic and phase3_ms == 0:
        raise Exception('Triphasic shape requires duration_phase3_ms > 0')
    elif shape != StimShape.Triphasic and phase3_ms != 0:
        raise Exception('duration_phase3_ms > 0 only allowed for Triphasic shape')

    first_period = delay_ms + phase1_ms + phase2_ms + phase3_ms + post_pulse_ms
    resolution = 1 / xdaq.sampleRate.rate * 1e3  # ms
    if first_period / resolution > 2**16:
        raise Exception(f'Stimulation period too long, max {2**16 * resolution:.2f} ms')
    if pre_ampsettle_ms > delay_ms:
        raise Exception('Pre amp settle out of range')
    if post_ampsettle_ms > post_pulse_ms:
        raise Exception('Post amp settle out of range')

    if post_charge_recovery_ms > post_pulse_ms:
        raise Exception('Post charge recovery out of range')
    if trigger == TriggerEvent.Level:
        if trigger_pol == TriggerPolarity.Low:
            trigger_pol = TriggerPolarity.High
        elif trigger_pol == TriggerPolarity.High:
            trigger_pol = TriggerPolarity.Low

    kwargs = {
        'stream': stream,
        'channel': channel,
        'polarity': polarity,
        'shape': shape,
        'delay_ms': delay_ms,
        'duration_phase1_ms': phase1_ms,
        'duration_phase2_ms': phase2_ms,
        'duration_phase3_ms': phase3_ms,
        'amp_neg_mA': amp_neg_mA,
        'amp_pos_mA': amp_pos_mA,
        'pre_ampsettle_ms': pre_ampsettle_ms,
        'post_ampsettle_ms': post_ampsettle_ms,
        'pulses': pulses,
        'duration_pulse_ms': post_pulse_ms,
        'trigger': trigger,
        'trigger_source': trigger_source,
        'trigger_pol': trigger_pol,
        'step_size': step_size,
        'post_charge_recovery_ms': post_charge_recovery_ms,
    }
    xdaq.stim.configure(**kwargs, enable=True)
    return lambda: xdaq.stim.configure(**kwargs, enable=False)


def pulses(mA: float, frequency: float):
    """
    Create a biphasic pulse
             |---------|                   |---------|
             |         |                   |         |
    ---------|         |         |---------|         |
                       |         |                   |
                       |---------|                   |---------
    |-delay--|                             |
             |-phase1--|                   |-phase1--| ...
                       |-phase2--|         |
                                 | post    |
                                   pulse   |

    |-----------------period---------------| = 1/frequency
    """
    phase_period_ms = 1e3 / frequency / 3
    return (lambda **kwargs: kwargs)(
        polarity=StartPolarity.cathodic if mA < 0 else StartPolarity.anodic,
        shape=StimShape.Biphasic,
        delay_ms=0,
        phase1_ms=phase_period_ms,
        phase2_ms=phase_period_ms,
        phase3_ms=0,
        amp_neg_mA=mA,
        amp_pos_mA=mA,
        pre_ampsettle_ms=0,
        post_ampsettle_ms=phase_period_ms,
        post_charge_recovery_ms=0,
        pulses=1,
        post_pulse_ms=phase_period_ms,
        trigger=TriggerEvent.Level,
        trigger_pol=TriggerPolarity.High,
        step_size=StimStepSize.StimStepSize10uA,
    )
