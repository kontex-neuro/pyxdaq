#%%
import time
from pyxdaq.xdaq import get_XDAQ, XDAQ
from pyxdaq.stim import enable_stim
from pyxdaq.constants import StimStepSize, StimShape, StartPolarity, TriggerEvent, TriggerPolarity

xdaq = get_XDAQ(rhs=True, bitfile='bitfiles/xsr7310a75.bit')

print(xdaq.ports)

#%%


def pluses(mA: float, frequency: float):
    """
    Create a basic pluse with duty cycle 50%, current can be positive or negative.
         |-----|     |-----|
         |     |     |     |
    -----|     |-----|     |-----

    |--period--| = 1/frequency
    """
    half_period_ms = 1e3 / frequency / 2
    return (lambda **kwargs: kwargs)(
        # Polarity of the first phase
        polarity=StartPolarity.cathodic if mA < 0 else StartPolarity.anodic,
        # Shape of the pluse
        shape=StimShape.Monophasic,
        # Delay between the trigger and the start of the pluse
        delay_ms=0,
        # Duration of the first phase, there are only one phase in Monophasic shape
        phase1_ms=half_period_ms,
        phase2_ms=0,
        phase3_ms=0,
        # Use 10uA step size, the current will be truncated to the nearest 10uA
        step_size=StimStepSize.StimStepSize10uA,
        # Amplitude of the first phase
        amp_phase1_mA=mA,
        amp_phase2_mA=0,
        # Please refer to Intan Manual for ampsettle and charge recovery
        pre_ampsettle_ms=0,
        post_ampsettle_ms=half_period_ms,
        post_charge_recovery_ms=0,
        # The duration after the pluse before the next pluse
        post_pluse_ms=half_period_ms,
        # Sending pluses continuously when the trigger is high
        trigger=TriggerEvent.Level,
        trigger_pol=TriggerPolarity.High,
        # Since we are using Level trigger, sending one pluse each time
        pulses=1,
    )


def send_pulses(xdaq: XDAQ, stream, channel, duration_ms, pluse_current_mA, pluse_frequency):
    software_trigger_id = 1

    # The enable_stim function will return a function to disable the stim
    disable_stim = enable_stim(
        xdaq=xdaq,
        stream=stream,
        channel=channel,
        # Trigger source, 24~31 is the software trigger 1~8
        trigger_source=24 + software_trigger_id - 1,
        **pluses(pluse_current_mA, pluse_frequency)
    )

    # Enable software trigger
    xdaq.manual_trigger(software_trigger_id, True)

    # Calculate the number of steps to run, the number of steps should be multiple of 128 to avoid alignment error
    run_steps = (int(duration_ms / 1000 * xdaq.sampleRate.rate) + 127) // 128 * 128

    # Set the maximum number of steps to run
    # this could be replaced by
    # xdaq.readDataBlock(run_steps).to_samples(),
    # But it will take longer time to run, since it will read data from FPGA
    xdaq.setMaxTimeStep(run_steps)
    xdaq.setContinuousRunMode(False)
    xdaq.run()
    while xdaq.is_running():
        time.sleep(0.1)

    # Disable software trigger after the run
    xdaq.manual_trigger(software_trigger_id, False)
    # Disable the stim, a stim can run multiple times, here we set the stim every time for demonstration
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
