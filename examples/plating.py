#%%
import math

from pyxdaq.constants import StartPolarity, StimShape, StimStepSize, TriggerEvent, TriggerPolarity
from pyxdaq.impedance import Frequency, Strategy
from pyxdaq.stim import enable_stim
from pyxdaq.xdaq import XDAQ, get_XDAQ

xdaq = get_XDAQ(rhs=True)

print(xdaq.ports)
# XDAQ supports up to 4 X3SR32 Headstages, each headstage has 2 streams and each stream has 16 channels
# HDMI Port 0: Stream 0 (ch 0-16), Stream 1 (ch 16-31)
# HDMI Port 1: Stream 2 (ch 0-16), Stream 3 (ch 16-31)
# HDMI Port 2: Stream 4 (ch 0-16), Stream 5 (ch 16-31)
# HDMI Port 3: Stream 6 (ch 0-16), Stream 7 (ch 16-31)
# The get_XDAQ function will connect to the XDAQ and detect the number of headstages connected

#%%


def find_step_size(target_na):
    """
    Find the best step size to reach the target current
    """
    best_error = float('inf')
    best = None
    for step_size in StimStepSize:
        if math.isnan(step_size.nA):
            continue
        steps = min(255, round(target_na / step_size.nA))  # 255 is the maximum current step
        error = abs(target_na - step_size.nA * steps)
        if error < best_error:
            best_error = error
            best = step_size
    return best


def create_monophasic_pulse(mA: float, frequency: float):
    """
    Create a basic pulse with duty cycle 50%, current can be positive or negative.
         |-----|     |-----|
         |     |     |     |
    -----|     |-----|     |-----

    |--period--| = 1/frequency
    """
    half_period_ms = 1e3 / frequency / 2
    return (lambda **kwargs: kwargs)(
        # Polarity of the first phase
        polarity=StartPolarity.cathodic if mA < 0 else StartPolarity.anodic,
        # Shape of the pulse
        shape=StimShape.Monophasic,
        # Delay between the trigger and the start of the pulse
        delay_ms=0,
        # Duration of the first phase, there are only one phase in Monophasic shape
        phase1_ms=half_period_ms,
        phase2_ms=0,
        phase3_ms=0,
        step_size=find_step_size(abs(mA) * 1e6),
        # Current of the positive and negative phase, both value should be positive
        amp_neg_mA=0 if mA > 0 else -mA,
        amp_pos_mA=mA if mA > 0 else 0,
        # Please refer to Intan Manual for ampsettle and charge recovery
        pre_ampsettle_ms=0,
        post_ampsettle_ms=half_period_ms,
        post_charge_recovery_ms=0,
        # The duration after the pulse before the next pulse
        post_pulse_ms=half_period_ms,
        # Sending pulses continuously when the trigger is high
        trigger=TriggerEvent.Level,
        trigger_pol=TriggerPolarity.High,
        # Since we are using Level trigger, sending one pulse each time
        pulses=1,
    )


# Swap out the pulses function with this one to send biphasic pulses
def create_biphasic_pulse(mA: float, frequency: float):
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
    period_ms = 1e3 / frequency
    return (lambda **kwargs: kwargs)(
        polarity=StartPolarity.cathodic if mA < 0 else StartPolarity.anodic,
        shape=StimShape.Biphasic,
        delay_ms=0,
        phase1_ms=period_ms / 3,
        phase2_ms=period_ms / 3,
        phase3_ms=0,
        step_size=find_step_size(abs(mA) * 1e6),
        amp_neg_mA=abs(mA),
        amp_pos_mA=abs(mA),
        pre_ampsettle_ms=0,
        post_ampsettle_ms=period_ms / 3,
        post_charge_recovery_ms=0,
        post_pulse_ms=period_ms / 3,
        trigger=TriggerEvent.Level,
        trigger_pol=TriggerPolarity.High,
        pulses=1,
    )


def send_pulses(
    xdaq: XDAQ,
    stream: int,
    channel: int,
    duration_ms: float,
    pulse_current_mA: float,
    pulse_frequency: float,
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
        **create_monophasic_pulse(pulse_current_mA, pulse_frequency)
    )

    # Enable software trigger
    xdaq.manual_trigger(software_trigger_id, True)

    # Calculate the number of steps to run, the number of steps should be multiple of 128 to avoid alignment error
    run_steps = (int(duration_ms / 1000 * xdaq.sampleRate.rate) + 127) // 128 * 128

    # Start running
    xdaq.setStimCmdMode(True)
    # Speed up the process by discarding the reading of the data block
    # It's possible to run without blocking the script, but this will require more complex logic
    # See runAndReadDataBlock function for more fine-grained control
    xdaq.runAndReadDataBlock(run_steps, discard=True)
    xdaq.setStimCmdMode(False)
    # Stop running

    # Disable software trigger after the run
    xdaq.manual_trigger(software_trigger_id, False)
    # Disable the stim, a stim can run multiple times, here we set the stim every time for demonstration
    disable_stim()

    return run_steps


target_stream = 0
target_channel = 0
current_mA = 0.06
for i in range(3):
    print(f'Run {i+1}: Checking impedance at 1000 Hz')
    magnitude1000, phase1000 = xdaq.measure_impedance(
        frequency=Frequency(1000),
        # 0.2 seconds per measurement
        strategy=Strategy.from_duration(0.2),
        channels=[target_channel],
        progress=False
    )
    print(f'Impedance at 1000 Hz: {magnitude1000[target_stream,0]:.2f} Ohm')
    print(f'Sending 50Hz {current_mA}mA pulses for 1 second (dutycycle 50%)')
    run_steps = send_pulses(
        xdaq,
        stream=target_stream,
        channel=target_channel,
        duration_ms=1000,
        pulse_current_mA=current_mA,
        pulse_frequency=50
    )
