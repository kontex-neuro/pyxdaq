import signal
import time
import numpy as np

from pyxdaq.datablock import DataBlock, amplifier2uv
from pyxdaq.xdaq import get_XDAQ
from pyxdaq.constants import RHS
from pyxdaq.stim import enable_stim, pulses


is_running = True
xdaq = get_XDAQ(rhs=True)
frame_size = xdaq.getSampleSizeBytes()
sample_rate = xdaq.sampleRate.rate
print(
    f"Frame size: {frame_size} bytes @ {sample_rate} Hz = "
    f"{frame_size * sample_rate / 1e6:.2f} MB/s"
)

xdaq.setContinuousRunMode(True)
num_streams = xdaq.numDataStream


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
    if length % frame_size != 0:
        if is_running:
            print(f"[Warning] invalid frame length {length}")
        else:
            # invalid frame length, could be the last data chunk.
            pass
        return

    block = DataBlock.from_buffer(xdaq.rhs, frame_size, buffer, num_streams)
    samples = block.to_samples()

    probe_channel = 1
    amp_uv = amplifier2uv(samples.amp[:, probe_channel, target_stream, 1])
    # samples.amp[:, 0, :, :] = Channel 0
    # samples.amp[:, 1, :, :] = Channel 1
    # samples.amp[:, :, :, 0] = DC low-gain amplifier
    # samples.amp[:, :, :, 1] = AC high-gain amplifier

    # Replace the following condition to trigger stimulation
    # Here is an example of triggering stimulation when the maximum amplitude exceeds 500 μV
    max_amp = np.max(np.abs(amp_uv))
    uv_threshold = 500

    if max_amp > uv_threshold:
        # Enable stimulation
        xdaq.dev.SetWireInValue(RHS.WireInManualTriggers, 0x1)
        xdaq.setStimCmdMode(True)

        print(f"Stim triggered at Timestep:{samples.ts[0]:8d}")

        # Disable stimulation
        xdaq.setStimCmdMode(False)
        xdaq.dev.SetWireInValue(RHS.WireInManualTriggers, 0x0)


# Enable stimulation needs to be done BEFORE acquisition
# Here is an example of enabling stimulation on stream 0, channel 0, with a 10 Hz pulse at 1 mA
target_stream = 0
stim_channel = 0

disable_stim = enable_stim(
    xdaq=xdaq,
    stream=target_stream,
    channel=stim_channel,
    trigger_source=24,
    **pulses(mA=1, frequency=10),
)

# Use the aligned-buffer context to start/stop the callback queue
with xdaq.dev.start_receiving_aligned_buffer(
    xdaq.ep.PipeOutData,
    frame_size,
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
