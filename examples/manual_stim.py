import signal
import time
import numpy as np

from pyxdaq.datablock import DataBlock, amplifier2uv
from pyxdaq.xdaq import get_XDAQ
from pyxdaq.constants import RHS
from pyxdaq.stim import enable_stim
from plot_stim import pulses


is_running = True


def _handle_sigint(sig, frame):
    """Catch Ctrl+C and tell the main loop to exit."""
    global is_running
    is_running = False


signal.signal(signal.SIGINT, _handle_sigint)
xdaq = get_XDAQ(rhs=True)
frame_size = xdaq.getSampleSizeBytes()
sample_rate = xdaq.sampleRate.rate
print(
    f"Frame size: {frame_size} bytes @ {sample_rate} Hz = "
    f"{frame_size * sample_rate / 1e6:.2f} MB/s"
)

xdaq.setContinuousRunMode(True)

# Performance tuning parameters
hardware_events_per_sec = 100
bytes_per_sec = frame_size * sample_rate
# You may pass any integer here; the driver will round to a valid power-of-two chunk
chunk_size = int(bytes_per_sec / hardware_events_per_sec)

total_bytes_received = 0
recent_events = []
num_streams = xdaq.numDataStream
start_time = time.time()

target_stream = 0
target_channel = 0
uv_threshold = 500


def on_data_received(data: bytes, error: str):
    """
    Called in a dedicated thread whenever a data frame arrives.

    NOTE: this callback holds the Python GIL. If you do heavy work here, the
    Python-side queue may back up (HW keeps running, but this queue grows).
    It's OK to compute here as long as it keep up with the target rate.

    CALLBACK LIFETIME: even after xdaq.stop(), this callback may still be
    invoked until exit the start_receiving_aligned_buffer context.
    """
    global total_bytes_received, recent_events, is_running, target_stream, target_channel, uv_threshold

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

    try:
        block = DataBlock.from_buffer(xdaq.rhs, frame_size, buffer, num_streams)
        samples = block.to_samples()

        amp_uv = amplifier2uv(samples.amp[:, 1, target_stream, 1])
        # Index breakdown:
        # - samples.amp[:, 0, :, :] = Channel 0
        # - samples.amp[:, 1, :, :] = Channel 1
        # - samples.amp[:, :, :, 0] = DC low-gain amplifier
        # - samples.amp[:, :, :, 1] = AC high-gain amplifier

        max_amp = np.max(np.abs(amp_uv))
        stim_triggered = False

        if max_amp > uv_threshold:
            xdaq.dev.SetWireInValue(RHS.WireInManualTriggers, 0x1)
            xdaq.setStimCmdMode(True)
            stim_triggered = True
            xdaq.setStimCmdMode(False)
            xdaq.dev.SetWireInValue(RHS.WireInManualTriggers, 0x0)

        # Update throughput stats
        total_bytes_received += length
        now = time.time()
        recent_events.append((length, now - start_time))
        if len(recent_events) > 200:
            recent_events.pop(0)

        elapsed = (now - start_time) or 1e9
        avg_rate = total_bytes_received / elapsed
        window = (recent_events[-1][1] - recent_events[0][1]) or 1e9
        recent_rate = sum(e[0] for e in recent_events) * 199 / 200 / window

        print(
            f"Chunk: {length:8d} B | "
            f"Timestep: {samples.ts[0]:8d} | "
            f"Recent: {recent_rate/1e6:5.2f} MB/s | "
            f"Avg: {avg_rate/1e6:5.2f} MB/s",
            f"Stim: {'ON' if stim_triggered else 'OFF'}",
            end="  \r",
        )
    except Exception as e:
        print(f"An exception occurred: {e}")
        is_running = False
        return


pulse_current_mA = 1
pulse_frequency = 10

disable_stim = enable_stim(
    xdaq=xdaq,
    stream=target_stream,
    channel=target_channel,
    trigger_source=24,
    **pulses(pulse_current_mA, pulse_frequency),
)

# Use the aligned-buffer context to start/stop the callback queue
with xdaq.dev.start_receiving_aligned_buffer(
    xdaq.ep.PipeOutData,
    frame_size,
    on_data_received,
    chunk_size=chunk_size,
):
    # Kick off acquisition
    xdaq.start(continuous=True)

    # Wait until SIGINT
    while is_running:
        time.sleep(0.1)

    # Stop acquisition
    xdaq.stop(wait=True)
    # Callback may still run until we exit this block

disable_stim()
print("\nExiting...")
