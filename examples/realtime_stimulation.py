import signal
import time
import numpy as np

from pyxdaq.datablock import amplifier2uv
from pyxdaq.xdaq import get_XDAQ
from pyxdaq.stim import enable_stim, pulses


xdaq = get_XDAQ(rhs=True)
is_running = True


def _handle_sigint(sig, frame):
    """Catch Ctrl+C and tell the main loop to exit."""
    global is_running
    is_running = False


signal.signal(signal.SIGINT, _handle_sigint)


def on_data_received(data: bytes, error: str):
    """
    Called in a dedicated thread whenever a data frame arrives.

    NOTE: this callback holds the Python GIL. If you do heavy work here, the
    Python-side queue may back up (HW keeps running, but this queue grows).
    It's OK to compute here as long as it keep up with the target rate.

    CALLBACK LIFETIME: even after xdaq.stop(), this callback may still be
    invoked until exit the start_receiving_aligned_buffer context.
    """

    if error:
        print(f"[XDAQ error] {error}")
        return

    if not data:
        return

    buffer = bytearray(data)
    length = len(buffer)
    # Error check: if not running, it could be the last data chunk.
    if not is_running:
        print(f"[Warning] invalid frame length {length}")
        return

    # Parse: convert buffer to samples
    samples = xdaq.buffer_to_samples(buffer)

    amp_uv = amplifier2uv(samples.amp[:, 1, target_stream, 1])
    # Shape: [n_samples, channels, datastreams]           for RHD;
    #        [n_samples, channels, datastreams, [DC, AC]] for RHS
    #   n_samples: number of samples
    #    channels: number of channels per datastream (32 for RHD, 16 for RHS)
    # datastreams: number of datastreams (depends on type and numbers of attached headstages)
    #    [DC, AC]: DC/AC amplifier channel (RHS only); 0: DC low-gain, 1: AC high-gain

    # Replace the following condition to trigger stimulation
    # Here is an example of triggering stimulation when the maximum amplitude exceeds 500 μV
    max_amp = np.max(np.abs(amp_uv))
    uv_threshold = 500

    if max_amp > uv_threshold:
        # Enable manual trigger
        xdaq.manual_trigger(0, True)
        print(f"Stim triggered at Timestep:{samples.ts[0]:8d}")
        # Disable manual trigger
        xdaq.manual_trigger(0, False)


# Enable stimulation needs to be done BEFORE acquisition
# Here is an example of enabling stimulation on stream 0, channel 0, with a 10 Hz pulse at 1 mA
target_stream = 0
# Call disable_stim at the end of action to prevent unintended stimulation
disable_stim = enable_stim(
    xdaq=xdaq,
    stream=target_stream,
    channel=0,
    trigger_source=24,
    **pulses(mA=1, frequency=10),
)

print("Starting XDAQ acquisition...")
# Use the aligned-buffer context to start/stop the callback queue
with xdaq.start_receiving_buffer(
    on_data_received,
):
    # Kick off acquisition
    xdaq.start(continuous=True)

    # Wait until SIGINT
    while is_running:
        time.sleep(0.1)

    # Stop acquisition
    xdaq.stop(wait=True)
    # Callback may still run until we exit this block

# Disable stimulation AFTER acquisition
disable_stim()
print("\nExiting...")
