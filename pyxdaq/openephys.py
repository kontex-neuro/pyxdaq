import json
import pathlib
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class OpenEphysMetadata:
    gui_version: str
    source_processor_name: str = "XDAQ"
    source_processor_id: int = 100
    experiment_index: int = 0
    recording_index: int = 0

    def get_stream_info(
        self, stream_name: str, sample_rate: float, num_channels: int, bit_volts: float
    ) -> dict:
        folder_name = f"{self.source_processor_name}-{self.source_processor_id}.{stream_name}"
        return {
            "stream_name":
                stream_name,
            "folder_name":
                folder_name,
            "sample_rate":
                float(sample_rate),
            "num_channels":
                num_channels,
            "source_processor_id":
                self.source_processor_id,
            "source_processor_name":
                self.source_processor_name,
            "channels":
                [{
                    "channel_name": f"C{i}",
                    "bit_volts": bit_volts
                } for i in range(num_channels)],
        }

    def write_structure_oebin(self, recording_path: pathlib.Path, streams: List[dict]) -> None:
        structure = {
            "GUI version": self.gui_version,
            "continuous": streams,
            "events": [],
            "spikes": [],
        }
        oebin_path = recording_path / "structure.oebin"
        with open(oebin_path, 'w') as f:
            json.dump(structure, f, indent=4)


@dataclass
class RecordingPaths:
    root: pathlib.Path
    session: pathlib.Path
    record_node: pathlib.Path
    experiment: Optional[pathlib.Path] = None
    recording: Optional[pathlib.Path] = None

    @classmethod
    def create(cls, root_path: str, record_node: str) -> "RecordingPaths":
        root = pathlib.Path(root_path)
        session_ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        session = root / session_ts
        record_node_path = session / record_node
        record_node_path.mkdir(parents=True, exist_ok=True)
        return cls(root=root, session=session, record_node=record_node_path)

    def new_experiment(self, experiment_index: int) -> pathlib.Path:
        self.experiment = self.record_node / f"experiment{experiment_index}"
        self.experiment.mkdir(exist_ok=True)
        return self.experiment

    def new_recording(self, recording_index: int) -> pathlib.Path:
        if self.experiment is None:
            raise ValueError("Cannot create recording path before experiment path.")
        self.recording = self.experiment / f"recording{recording_index}"
        self.recording.mkdir(exist_ok=True)
        return self.recording
