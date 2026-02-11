import signal
import time

from pyxdaq.xdaq import get_XDAQ
from pyxdaq.writer import OpenEphysWriter

is_running = True
ttl_is_on = False
ttl_period = 20 * 0.001  # 20 ms
ttl_start_time = time.time()


def _handle_sigint(sig, frame):
    """Catch Ctrl+C and tell the main loop to exit."""
    global is_running
    is_running = False


signal.signal(signal.SIGINT, _handle_sigint)


def on_error(error: str):
    """Prints errors from the data stream and stops the application."""
    global is_running
    if not is_running:  # Already stopping, ignore further errors
        return
    print(f"\n[Callback Error] {error}")
    is_running = False


xdaq = get_XDAQ()

with OpenEphysWriter(xdaq, root_path=".") as writer:
    with xdaq.start_receiving_samples(callbacks=[writer.write_data], on_error=on_error):
        # Kick off acquisition
        xdaq.start(continuous=True)

        while is_running:
            if time.time() - ttl_start_time >= ttl_period:
                ttl_is_on = not ttl_is_on
                xdaq.setTTLout('all', ttl_is_on)
                ttl_start_time = time.time()
            time.sleep(0.001)

        # Stop acquisition
        writer.stop_recording()
        xdaq.stop(wait=True)
        # Callback may still be invoked until we exit the context manager

print("\nExiting...")
