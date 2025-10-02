import logging
from typing import Dict

from .datablock import Samples
from .openephys import OpenEphysMetadata, RecordingPaths
from .stream import (DeviceType, RHDStreamer, RHSStreamer, StreamWriter)
from .xdaq import XDAQ

logger = logging.getLogger(__name__)


class OpenEphysWriter:

    def __init__(
        self,
        xdaq: XDAQ,
        root_path: str,
        device_type: DeviceType,
        record_node: str = "Record Node 101",
        gui_version: str = "0.6.4"
    ):
        self.xdaq = xdaq
        self.paths = RecordingPaths.create(root_path, record_node)
        self.metadata = OpenEphysMetadata(gui_version=gui_version)

        match device_type:
            case DeviceType.RHS:
                self.streamer = RHSStreamer()
            case DeviceType.RHD:
                self.streamer = RHDStreamer()
            case _:
                raise ValueError(f"Unsupported device type: {device_type}")

        self._is_recording = False
        self.stream_writers: Dict[str, StreamWriter] = {}
        self.sample_rate = 0

    def __enter__(self):
        self.start_recording()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_recording()

    def start_recording(self):
        if not self.paths.experiment:
            self.metadata.experiment_index += 1
            self.metadata.recording_index = 0
            self.paths.new_experiment(self.metadata.experiment_index)

        self.metadata.recording_index += 1
        recording_path = self.paths.new_recording(self.metadata.recording_index)

        self.sample_rate = self.xdaq.sampleRate.rate

        stream_configs = self.streamer.create_stream_configs()
        stream_infos = []

        for key, config in stream_configs.items():
            folder_name = (
                f"{self.metadata.source_processor_name}-"
                f"{self.metadata.source_processor_id}.{config.stream_name}"
            )
            stream_path = recording_path / "continuous" / folder_name
            writer = StreamWriter(
                stream_config=config,
                stream_path=stream_path,
                streamer=self.streamer,
                sample_rate=self.sample_rate
            )
            writer.open()
            self.stream_writers[key] = writer
            logger.info(f"Started recording in: {stream_path}")

            stream_info = self.metadata.get_stream_info(
                stream_name=config.stream_name,
                sample_rate=self.sample_rate,
                num_channels=self.xdaq.numDataStream * self.streamer.num_channels(),
                bit_volts=config.bit_volts
            )
            stream_infos.append(stream_info)

        self.metadata.write_structure_oebin(recording_path, stream_infos)
        self._is_recording = True

    def stop_recording(self):
        if not self._is_recording:
            return

        for writer in self.stream_writers.values():
            writer.close()
            logger.info(f"Stopped recording. Data saved in: {writer.stream_path}")

        self.stream_writers.clear()
        self._is_recording = False

    def write_data(self, samples: "Samples"):
        if not self._is_recording:
            return
        if samples.n == 0:
            return

        for writer in self.stream_writers.values():
            writer.write_sample_data(samples)
