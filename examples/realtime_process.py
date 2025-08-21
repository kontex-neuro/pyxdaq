import signal
import time

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

bytes_per_sec = frame_size * sample_rate

total_bytes_received = 0
recent_events = []
num_streams = xdaq.numDataStream
start_time = time.time()

current_error = None


def on_data_received(data: bytes, error: str):
    """
    Called in a dedicated thread whenever a data frame arrives.

    NOTE: this callback holds the Python GIL. If you do heavy work here, the
    Python-side queue may back up (HW keeps running, but this queue grows).
    It's OK to compute here as long as it keep up with the target rate.

    CALLBACK LIFETIME: even after xdaq.stop(), this callback may still be
    invoked until exit the start_receiving_buffer context.
    """
    global total_bytes_received, recent_events, current_error

    if error:
        current_error = error
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
    try:
        samples = xdaq.buffer_to_samples(buffer)
    except ValueError as e:
        current_error = str(e)
        return

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
        f"Chunk: {length:8d} B"
        f" | Sample Index: {samples.sample_index[0]:8d}"
        f" | Recent: {recent_rate/1e6:5.2f} MB/s"
        f" | Avg: {avg_rate/1e6:5.2f} MB/s" + (
            f" | Timestep: {samples.timestamp[0]/1e6:8.3f} s "
            if samples.timestamp is not None else ""
        ),
        end="  \r",
        flush=True,  # Remove this in actual experiments
    )


# Start receiving data
with xdaq.start_receiving_buffer(on_data_received):
    # Kick off acquisition
    xdaq.start(continuous=True)

    # Wait until interrupted or error occurs
    while is_running:
        time.sleep(0.01)
        if current_error is not None:
            print(f"\n[Callback Error] {current_error}")
            # Stop callback from processing more data
            is_running = False

    # Stop acquisition
    xdaq.stop(wait=True)
    # Callback may still run until we exit this block

print("\nExiting...")
