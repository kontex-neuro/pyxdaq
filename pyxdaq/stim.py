from .constants import StartPolarity, StimShape, StimStepSize, TriggerEvent, TriggerPolarity
from .xdaq import XDAQ


def enable_stim(
    xdaq: XDAQ,
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
    xdaq.set_stim(**kwargs, enable=True)
    return lambda: xdaq.set_stim(**kwargs, enable=False)
