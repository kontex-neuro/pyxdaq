import signal
import time
import numpy as np
from contextlib import contextmanager

from pyxdaq.datablock import DataBlock, amplifier2uv
from pyxdaq.xdaq import get_XDAQ
from pyxdaq.constants import RHS
from pyxdaq.stim import enable_stim, pulses


class StimController:
    def __init__(self, stream: int, channel: int, mA: float, frequency: float):
        self.xdaq = get_XDAQ(rhs=True)
        self.is_running = True

        self.stream = stream
        self.channel = channel
        self.mA = mA
        self.frequency = frequency

        self.frame_size = self.xdaq.getSampleSizeBytes()
        self.num_streams = self.xdaq.numDataStream
        signal.signal(signal.SIGINT, self._handle_sigint)

        print(
            f"Enabling stimulation on stream {self.stream}, channel {self.channel}, {self.mA} mA, {self.frequency} Hz"
        )

        self.disable_stim = enable_stim(
            xdaq=self.xdaq,
            stream=self.stream,
            channel=self.channel,
            trigger_source=24,
            **pulses(mA=self.mA, frequency=self.frequency),
        )

    def _handle_sigint(self, sig, frame):
        """
        Catch Ctrl+C and tell the main loop to exit.
        """
        self.is_running = False

    def _set_manual_stim_trigger(self, trigger_on: bool):
        self.xdaq.dev.SetWireInValue(
            RHS.WireInManualTriggers, 0x1 if trigger_on else 0x0
        )
        self.xdaq.setStimCmdMode(trigger_on)

    def _trigger_stim(self):
        """
        Manually trigger stimulation.
        """
        self._set_manual_stim_trigger(True)

        # Here is an example of printing the timestamp of the stimulation trigger
        print(f"Stimulation triggered at Timestep:{self.samples.ts[0]:8d}")

        self._set_manual_stim_trigger(False)

    def _on_data_received(self, data: bytes, error: str):
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
        if length % self.frame_size != 0:
            if self.is_running:
                print(f"[Warning] invalid frame length {length}")
            else:
                # invalid frame length, could be the last data chunk.
                pass
            return

        block = DataBlock.from_buffer(
            self.xdaq.rhs, self.frame_size, buffer, self.num_streams
        )
        self.samples = block.to_samples()

        # Check amplitude on probe channel 1
        probe_channel = 1
        amp_uv = amplifier2uv(self.samples.amp[:, probe_channel, self.stream, 1])
        # samples.amp[:, 0, :, :] = Channel 0
        # samples.amp[:, 1, :, :] = Channel 1
        # samples.amp[:, :, :, 0] = DC low-gain amplifier
        # samples.amp[:, :, :, 1] = AC high-gain amplifier
        max_amp = np.max(np.abs(amp_uv))

        # Here is an example of triggering stimulation when the maximum amplitude exceeds 500 μV
        if max_amp > 500:
            self._trigger_stim()

    def run(self):
        """
        Run continuous data acquisition with stimulation triggering.
        """
        sample_rate = self.xdaq.sampleRate.rate

        print(
            f"Frame size: {self.frame_size} bytes @ {sample_rate} Hz = "
            f"{self.frame_size * sample_rate / 1e6:.2f} MB/s"
        )

        self.xdaq.setContinuousRunMode(True)

        hardware_events_per_sec = 100
        chunk_size = int(self.frame_size * sample_rate / hardware_events_per_sec)

        # Use the aligned-buffer context to start/stop the callback queue
        with self.xdaq.dev.start_receiving_aligned_buffer(
            self.xdaq.ep.PipeOutData,
            self.frame_size,
            self._on_data_received,
            chunk_size=chunk_size,
        ):
            # Kick off acquisition
            self.xdaq.start(continuous=True)

            # Wait until SIGINT
            while self.is_running:
                time.sleep(0.1)

            # Stop acquisition
            self.xdaq.stop(wait=True)
            # Callback may still run until we exit this block


@contextmanager
def stimulation(stream: int, channel: int, mA: float, frequency: float):
    controller = StimController(stream, channel, mA, frequency)

    try:
        yield controller
    finally:
        print("Disabling stimulation")
        controller.disable_stim()
        print("\nExiting...")


if __name__ == "__main__":
    # Here is an example of enabling stimulation on stream 0, channel 0, with a 10 Hz pulse at 1 mA
    with stimulation(stream=0, channel=0, mA=1.0, frequency=10.0) as stim:
        stim.run()
