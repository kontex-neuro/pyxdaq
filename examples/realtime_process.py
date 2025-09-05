import signal
import time

from pyxdaq.datablock import Samples
from pyxdaq.xdaq import get_XDAQ
from pyxdaq.writer import OpenEphysWriter

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


def on_error(error: str):
    """Prints errors from the data stream and stops the application."""
    global is_running
    print(f"\n[Callback Error] {error}")
    is_running = False


def on_samples_received(samples: Samples):
    """
    Called in a dedicated thread whenever a data frame arrives.

    NOTE: this callback holds the Python GIL. If you do heavy work here, the
    Python-side queue may back up (HW keeps running, but this queue grows).
    It's OK to compute here as long as it keep up with the target rate.

    CALLBACK LIFETIME: even after xdaq.stop(), this callback may still be
    invoked until exit the start_receiving_buffer context.
    """
    global total_bytes_received, recent_events

    # The new API provides Samples objects directly.
    # We can calculate the byte length from the number of samples.
    length = samples.n * frame_size

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


# Start receiving data using the new samples-based API
# The OpenEphysWriter is used as a context manager to handle file operations.
with OpenEphysWriter(xdaq, root_path=".") as writer:
    # We can pass multiple callbacks; one for printing stats and one for writing data.
    with xdaq.start_receiving_samples(callbacks=[on_samples_received, writer.write_data],
                                      on_error=on_error):
        # Kick off acquisition
        xdaq.start(continuous=True)

        # Wait until interrupted or error occurs
        while is_running:
            time.sleep(0.01)

        # Stop acquisition
        xdaq.stop(wait=True)
        # Callback may still run until we exit this block

print("\nExiting...")
