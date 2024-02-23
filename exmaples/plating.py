#%%
import time
from pyxdaq.xdaq import get_XDAQ, XDAQ
from pyxdaq.stim import enable_stim
from pyxdaq.constants import StimStepSize, StimShape, StartPolarity, TriggerEvent, TriggerPolarity
from pyxdaq.impedance import TestFrequency

xdaq = get_XDAQ(rhs=True)

print(xdaq.ports)
# XDAQ supports up to 4 X3SR32 Headstages, each headstage has 2 streams and each stream has 16 channels
# HDMI Port 0: Stream 0 (ch 0-16), Stream 1 (ch 16-31)
# HDMI Port 1: Stream 2 (ch 0-16), Stream 3 (ch 16-31)
# HDMI Port 2: Stream 4 (ch 0-16), Stream 5 (ch 16-31)
# HDMI Port 3: Stream 6 (ch 0-16), Stream 7 (ch 16-31)
# The get_XDAQ function will connect to the XDAQ and detect the number of headstages connected

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
        # Current of the positive and negative phase, both value should be positive
        amp_neg_mA=0 if mA > 0 else -mA,
        amp_pos_mA=mA if mA > 0 else 0,
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


# Swap out the pluses function with this one to send biphasic pluses
def biphasic_pluses(mA: float, frequency: float):
    """
    Create a biphasic pluse
             |---------|                   |---------|
             |         |                   |         |
    ---------|         |         |---------|         |
                       |         |                   |
                       |---------|                   |---------
    |-delay--|                             |
             |-phase1--|                   |-phase1--| ...
                       |-phase2--|         |
                                 | post    |
                                   pluse   |

    |-----------------period---------------| = 1/frequency
    """
    period_ms = 1e3 / frequency
    return (lambda **kwargs: kwargs)(
        polarity=StartPolarity.cathodic if mA < 0 else StartPolarity.anodic,
        shape=StimShape.Biphasic,
        delay_ms=0,
        phase1_ms=period_ms / 3,
        phase2_ms=period_ms / 3,
        phase3_ms=0,
        step_size=StimStepSize.StimStepSize10uA,
        amp_neg_mA=abs(mA),
        amp_pos_mA=abs(mA),
        pre_ampsettle_ms=0,
        post_ampsettle_ms=period_ms / 3,
        post_charge_recovery_ms=0,
        post_pluse_ms=period_ms / 3,
        trigger=TriggerEvent.Level,
        trigger_pol=TriggerPolarity.High,
        pulses=1,
    )


def send_pulses(
    xdaq: XDAQ,
    stream: int,
    channel: int,
    duration_ms: float,
    pluse_current_mA: float,
    pluse_frequency: float,
):
    # The software trigger id, 0~7, can be shared by multiple stim
    software_trigger_id = 0

    # The enable_stim function will return a function to disable the stim
    disable_stim = enable_stim(
        xdaq=xdaq,
        stream=stream,
        channel=channel,
        # Trigger source, 24~31 is the software trigger 0~7
        trigger_source=24 + software_trigger_id,
        **pluses(pluse_current_mA, pluse_frequency)
    )

    # Enable software trigger
    xdaq.manual_trigger(software_trigger_id, True)

    # Calculate the number of steps to run, the number of steps should be multiple of 128 to avoid alignment error
    run_steps = (int(duration_ms / 1000 * xdaq.sampleRate.rate) + 127) // 128 * 128

    # Set the maximum number of steps to run and start running.
    # Start running
    xdaq.setMaxTimeStep(run_steps)
    xdaq.setContinuousRunMode(False)
    xdaq.run()
    while xdaq.is_running():
        time.sleep(0.1)
    # Stop running
    # The code between Start running and Stop running can be replaced by
    # xdaq.readDataBlock(run_steps).to_samples()
    # But it will take slightly longer time to run, since it will read data from FPGA

    # Disable software trigger after the run
    xdaq.manual_trigger(software_trigger_id, False)
    # Disable the stim, a stim can run multiple times, here we set the stim every time for demonstration
    disable_stim()

    return run_steps


target_stream = 0
target_channel = 0
for i in range(3):
    print(f'Run {i+1}: Checking impedance at 1000 Hz')
    magnitude1000, phase1000 = xdaq.measure_impedance(
        test_frequency=TestFrequency(1000), channels=[target_channel], progress=False
    )
    print(f'Impedance at 1000 Hz: {magnitude1000[target_stream,0]:.2f} Ohm')
    print(f'Sending 50Hz 1mA pluses for 1 second (dutycycle 50%)')
    run_steps = send_pulses(
        xdaq,
        stream=target_stream,
        channel=target_channel,
        duration_ms=1000,
        pluse_current_mA=2,
        pluse_frequency=50
    )
