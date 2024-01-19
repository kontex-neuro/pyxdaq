#%%
import time
from pyxdaq.xdaq import get_XDAQ, XDAQ
from pyxdaq.stim import enable_stim
from pyxdaq.constants import StimStepSize, StimShape, StartPolarity, TriggerEvent, TriggerPolarity

xdaq = get_XDAQ(rhs=True, bitfile='bitfiles/xsr7310a75.bit')

print(xdaq.ports)

#%%


def pluses(mA: float, frequency: float):
    half_period_ms = 1e3 / frequency / 2
    return (lambda **kwargs: kwargs)(
        polarity=StartPolarity.cathodic if mA < 0 else StartPolarity.anodic,
        shape=StimShape.Monophasic,
        delay_ms=0,
        phase1_ms=half_period_ms,
        phase2_ms=0,
        phase3_ms=0,
        amp_phase1_mA=mA,
        amp_phase2_mA=0,
        pre_ampsettle_ms=0,
        post_ampsettle_ms=half_period_ms,
        post_charge_recovery_ms=0,
        pulses=1,
        post_pluse_ms=half_period_ms,
        trigger=TriggerEvent.Level,
        trigger_pol=TriggerPolarity.High,
        step_size=StimStepSize.StimStepSize10uA,
    )


def send_pulses(xdaq: XDAQ, stream, channel, duration_ms, pluse_current_mA, pluse_frequency):
    software_trigger_id = 1
    disable_stim = enable_stim(
        xdaq=xdaq,
        stream=stream,
        channel=channel,
        trigger_source=24 + software_trigger_id - 1,
        **pluses(pluse_current_mA, pluse_frequency)
    )
    xdaq.manual_trigger(software_trigger_id, True)
    run_steps = (int(duration_ms / 1000 * xdaq.sampleRate.rate) + 127) // 128 * 128
    xdaq.setMaxTimeStep(run_steps)
    xdaq.setContinuousRunMode(False)
    xdaq.run()
    while xdaq.is_running():
        time.sleep(0.1)
    xdaq.manual_trigger(software_trigger_id, False)
    disable_stim()
    return run_steps


#%%
target_stream = 0
target_channel = 0
for i in range(3):
    print(f'Run {i+1}: Checking impedance at 1000 Hz')
    magnitude1000, phase1000 = xdaq.measure_impedance(
        desired_test_frequency=1000, channels=[target_channel], progress=False
    )
    print(f'Impedance at 1000 Hz: {magnitude1000[target_stream,0]:.2f} Ohm')
    print(f'Sending 50Hz 1mA pluses for 1 second (dutycycle 50%)')
    run_steps = send_pulses(
        xdaq,
        stream=target_stream,
        channel=target_channel,
        duration_ms=1000,
        pluse_current_mA=1,
        pluse_frequency=50
    )
