import numpy as np

from pyxdaq.constants import *
from pyxdaq.datablock import Samples, adc2v
from pyxdaq.stim import enable_stim
from pyxdaq.xdaq import XDAQ, get_XDAQ

xdaq = get_XDAQ(rhs=True)

print(xdaq.ports)


def pulses(mA: float, frequency: float):
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


def send_pulses(
    xdaq: XDAQ, stream, channel, duration_ms, pulse_current_mA, pulse_frequency
) -> Samples:
    disable_stim = enable_stim(
        xdaq=xdaq,
        stream=stream,
        channel=channel,
        trigger_source=24,
        **pulses(pulse_current_mA, pulse_frequency)
    )
    xdaq.dev.SetWireInValue(RHS.WireInManualTriggers, 0x1)
    run_steps = (int(duration_ms / 1000 * xdaq.sampleRate.rate) + 127) // 128 * 128
    xdaq.setStimCmdMode(True)
    samples = xdaq.runAndReadDataBlock(run_steps).to_samples()
    xdaq.setStimCmdMode(False)
    xdaq.dev.SetWireInValue(RHS.WireInManualTriggers, 0x0)
    disable_stim()
    return samples


def plot_stim(sps: Samples, target_channel: int, target_stream: int):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(sps.amp.shape[0]) / xdaq.sampleRate.rate
    ax.plot(x, adc2v(sps.amp[:, target_channel, target_stream, 1]), c='b', alpha=0.4, label='amp')
    ax = ax.twinx()
    stim_on = (sps.stim[:, 0, target_stream] >> target_channel) & 1
    ax.plot(x, stim_on, c='r', label='stim', lw=4, alpha=0.5)
    ax.plot(
        x, (sps.stim[:, 2, target_stream] >> target_channel) & 1,
        c='lime',
        alpha=0.8,
        label='settle'
    )
    ax.plot(
        x, (sps.stim[:, 3, target_stream] >> target_channel) & 1,
        c='orange',
        alpha=0.8,
        label='recovery'
    )
    ax.plot(
        x,
        stim_on * (((sps.stim[:, 1, 0] >> target_channel) & 1).astype(np.int8) * 2 - 1),
        c='g',
        alpha=0.8,
        label='polarity'
    )
    ax.legend()
    ax.set_ylim(-1.5, 2)
    ax.set_yticks([-1, 0, 1])
    fig.tight_layout()
    plt.show()


target_stream = 0
target_channel = 0

samples = send_pulses(
    xdaq,
    stream=target_stream,
    channel=target_channel,
    duration_ms=1000,
    pulse_current_mA=1,
    pulse_frequency=10
)

plot_stim(samples, target_channel, target_stream)
