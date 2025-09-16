import gc
import json
import pathlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import BinaryIO, Dict, List, Optional

import numpy as np

from .datablock import Samples
from .xdaq import XDAQ


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
                [
                    {
                        "channel_name": f"CH{i + 1}",
                        "bit_volts": bit_volts
                    } for i in range(num_channels)
                ],
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
class StreamConfig:
    name: str  # 'AC', 'DC', or 'continuous'
    stream_name: str
    bit_volts: float
    offset: int

    BIT_VOLTS_AC: float = 0.195
    BIT_VOLTS_DC: float = -19.23
    OFFSET_AC: int = 32768
    OFFSET_DC: int = 512

    @classmethod
    def create_stream_configs(cls, is_rhs: bool) -> Dict[str, 'StreamConfig']:
        if is_rhs:
            return {
                'AC': cls('AC', 'AC', cls.BIT_VOLTS_AC, cls.OFFSET_AC),
                'DC': cls('DC', 'DC', cls.BIT_VOLTS_DC, cls.OFFSET_DC)
            }
        else:
            return {'continuous': cls('continuous', 'Rhythm_Data', cls.BIT_VOLTS_AC, cls.OFFSET_AC)}


@dataclass
class StreamWriter:
    stream_config: StreamConfig
    recording_path: pathlib.Path
    source_processor_name: str
    source_processor_id: int
    memmap_chunk_size: int

    stream_path: pathlib.Path = field(init=False)
    data_file: Optional[BinaryIO] = None
    sample_count: int = 0
    memmap_capacity: int = 0

    _memmap_paths: Dict[str, pathlib.Path] = field(default_factory=dict)
    _memmap_file_handles: Dict[str, BinaryIO] = field(default_factory=dict)
    _memmaps: Dict[str, np.memmap] = field(default_factory=dict)

    DTYPE_SAMPLE_NUMBERS: np.dtype = np.dtype(np.int64)
    DTYPE_TIMESTAMPS: np.dtype = np.dtype(np.float64)

    def __post_init__(self):
        folder_name = f"{self.source_processor_name}-{self.source_processor_id}.{self.stream_config.stream_name}"
        self.stream_path = self.recording_path / "continuous" / folder_name
        self.stream_path.mkdir(parents=True, exist_ok=True)
        self.memmap_capacity = self.memmap_chunk_size

        self._memmap_paths = {
            'sample_numbers': self.stream_path / "sample_numbers.mmap",
            'timestamps': self.stream_path / "timestamps.mmap",
        }

    def open(self):
        self.data_file = open(self.stream_path / "continuous.dat", 'ab')
        self._create_memmap('sample_numbers', self.DTYPE_SAMPLE_NUMBERS)
        self._create_memmap('timestamps', self.DTYPE_TIMESTAMPS)
        return self.stream_path

    def _create_memmap(self, name: str, dtype: np.dtype):
        fp = open(self._memmap_paths[name], 'w+b')
        fp.truncate(self.memmap_capacity * dtype.itemsize)
        self._memmap_file_handles[name] = fp
        self._memmaps[name] = np.memmap(
            fp, dtype=dtype, mode='r+', shape=(self.memmap_capacity,)
        )

    def _resize_memmaps(self, required_capacity: int):
        new_capacity = self.memmap_capacity
        while new_capacity < required_capacity:
            new_capacity += self.memmap_chunk_size

        for memmap in self._memmaps.values():
            memmap.flush()
        self._memmaps.clear()

        for name, dtype in [('sample_numbers', self.DTYPE_SAMPLE_NUMBERS),
                            ('timestamps', self.DTYPE_TIMESTAMPS)]:
            fp = self._memmap_file_handles[name]
            fp.truncate(new_capacity * dtype.itemsize)
            self._memmaps[name] = np.memmap(
                fp, dtype=dtype, mode='r+', shape=(new_capacity,)
            )

        self.memmap_capacity = new_capacity

    def write_sample_data(self, samples: "Samples", n_samples: int):
        required_capacity = self.sample_count + n_samples
        if required_capacity > self.memmap_capacity:
            self._resize_memmaps(required_capacity)

        offset = self.sample_count
        self._memmaps['sample_numbers'][offset:offset + n_samples] = samples.sample_index
        self._memmaps['timestamps'][offset:offset + n_samples] = samples.timestamp.astype(
            np.float64
        ) / 1_000_000.0
        self.sample_count += n_samples

    def write_amp_data(self, amp_data: np.ndarray, n_samples: int):
        if self.stream_config.name == 'DC':
            reshaped = amp_data.reshape(n_samples, -1)
            dc_10bit = reshaped & 0x3FE
            data_int16 = (dc_10bit.astype(np.int32) - self.stream_config.offset).astype(np.int16)
        else:  # AC or continuous
            reshaped = amp_data.reshape(n_samples, -1)
            data_int16 = (reshaped.astype(np.int32) - self.stream_config.offset).astype(np.int16)
        self.data_file.write(data_int16.tobytes())

    def close(self):
        if self.data_file:
            self.data_file.close()

        for memmap in self._memmaps.values():
            memmap.flush()
        gc.collect()
        self._memmaps.clear()

        if self.sample_count > 0:
            self._save_memmap_to_npy('sample_numbers', self.DTYPE_SAMPLE_NUMBERS)
            self._save_memmap_to_npy('timestamps', self.DTYPE_TIMESTAMPS)

        for handle in self._memmap_file_handles.values():
            handle.close()

        # Don't attempt to delete .mmap files, just note that they are temporary.
        for path in self._memmap_paths.values():
            if path.exists():
                print(f"Note: Temporary file {path} can be safely deleted manually.")

    def _save_memmap_to_npy(self, name: str, dtype: np.dtype):
        path = self._memmap_paths[name]
        if path.exists():
            try:
                with open(path, 'rb') as f:
                    data = np.fromfile(f, dtype=dtype, count=self.sample_count)
                np.save(self.stream_path / f"{name}.npy", data)
            except Exception as e:
                print(f"Warning: Error saving {name}.npy: {e}")


class OpenEphysWriter:

    def __init__(
        self,
        xdaq: "XDAQ",
        root_path: str,
        record_node: str = "Record Node 101",
        gui_version: str = "0.6.4"
    ):
        self.xdaq = xdaq
        self.root_path = pathlib.Path(root_path)
        session_ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.session_path = self.root_path / session_ts
        self.record_node_path = self.session_path / record_node
        self.record_node_path.mkdir(parents=True, exist_ok=True)

        self.metadata = OpenEphysMetadata(gui_version=gui_version)
        self.experiment_path: Optional[pathlib.Path] = None
        self.recording_path: Optional[pathlib.Path] = None

        self._is_recording = False
        self.stream_writers: Dict[str, StreamWriter] = {}
        self.sample_rate = 0

    def __enter__(self):
        self.start_recording()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_recording()

    def start_new_experiment(self):
        self.metadata.experiment_index += 1
        self.metadata.recording_index = 0
        self.experiment_path = self.record_node_path / f"experiment{self.metadata.experiment_index}"
        self.experiment_path.mkdir(exist_ok=True)

    def start_recording(self):
        if not self.experiment_path:
            self.start_new_experiment()

        self.metadata.recording_index += 1
        self.recording_path = self.experiment_path / f"recording{self.metadata.recording_index}"
        self.recording_path.mkdir(exist_ok=True)

        self.sample_rate = self.xdaq.sampleRate.rate
        memmap_chunk_size = self.sample_rate * 60

        stream_configs = StreamConfig.create_stream_configs(self.xdaq.rhs)
        stream_infos = []

        for key, config in stream_configs.items():
            writer = StreamWriter(
                stream_config=config,
                recording_path=self.recording_path,
                source_processor_name=self.metadata.source_processor_name,
                source_processor_id=self.metadata.source_processor_id,
                memmap_chunk_size=memmap_chunk_size
            )
            stream_path = writer.open()
            self.stream_writers[key] = writer
            print(f"Started recording in: {stream_path}")

            num_channels = self.xdaq.numDataStream * (16 if self.xdaq.rhs else 32)
            stream_info = self.metadata.get_stream_info(
                stream_name=config.stream_name,
                sample_rate=self.sample_rate,
                num_channels=num_channels,
                bit_volts=config.bit_volts
            )
            stream_infos.append(stream_info)

        self.metadata.write_structure_oebin(self.recording_path, stream_infos)
        self._is_recording = True

    def stop_recording(self):
        if not self._is_recording:
            return

        for writer in self.stream_writers.values():
            writer.close()
            print(f"Stopped recording. Data saved in: {writer.stream_path}")

        self.stream_writers.clear()
        self._is_recording = False

    def write_data(self, samples: "Samples"):
        if not self._is_recording:
            return
        n_samples = samples.n
        if n_samples == 0:
            return

        for writer in self.stream_writers.values():
            writer.write_sample_data(samples, n_samples)

        if self.xdaq.rhs:
            self.stream_writers['AC'].write_amp_data(samples.amp[..., 1], n_samples)
            self.stream_writers['DC'].write_amp_data(samples.amp[..., 0], n_samples)
        else:
            self.stream_writers['continuous'].write_amp_data(samples.amp, n_samples)
