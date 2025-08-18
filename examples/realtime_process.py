import signal
import time

from pyxdaq.datablock import DataBlock
from pyxdaq.xdaq import get_XDAQ

is_running = True


def _handle_sigint(sig, frame):
    """Catch Ctrl+C and tell the main loop to exit."""
    global is_running
    is_running = False


signal.signal(signal.SIGINT, _handle_sigint)

xdaq = get_XDAQ()

# Enable all streams even if no headstages are attached
# Remove this line for doing a real experiment
xdaq.enableDataStream("all", True)

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


def on_data_received(data: bytes, error: str):
    """
    Called in a dedicated thread whenever a data frame arrives.

    NOTE: this callback holds the Python GIL. If you do heavy work here, the
    Python-side queue may back up (HW keeps running, but this queue grows).
    It's OK to compute here as long as it keep up with the target rate.

    CALLBACK LIFETIME: even after xdaq.stop(), this callback may still be
    invoked until exit the start_receiving_aligned_buffer context.
    """
    global total_bytes_received, recent_events

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

    # Parse: convert bytes → samples
    block = DataBlock.from_buffer(xdaq.rhs, frame_size, buffer, num_streams)
    samples = block.to_samples()

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
        end="  \r",
        flush=True,  # Remove this in actual experiments
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

print("\nExiting...")
