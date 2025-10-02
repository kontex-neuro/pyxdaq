import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional
from abc import ABC, abstractmethod
from enum import Enum, auto

import numpy as np

from .datablock import Samples


class DeviceType(Enum):
    RHD = auto()
    RHS = auto()


@dataclass
class MemmapInfo:
    path: pathlib.Path
    dtype: np.dtype
    memmap: Optional[np.memmap] = None


@dataclass
class StreamConfig:
    name: str  # 'AC', 'DC', or 'continuous'
    stream_name: str
    bit_volts: float
    offset: int

    BIT_VOLTS_AC: float = 0.195  # uV
    BIT_VOLTS_DC: float = -19.23  # mV
    OFFSET_AC: int = 32768
    OFFSET_DC: int = 512


class DeviceStreamer(ABC):

    @abstractmethod
    def create_stream_configs(self) -> Dict[str, StreamConfig]:
        """Creates stream configurations for the device."""

    @abstractmethod
    def get_amp_data_for_stream(self, stream_name: str, samples: "Samples") -> np.ndarray:
        """Extracts amplifier data for a specific stream from a Samples object."""


class RHDStreamer(DeviceStreamer):

    def create_stream_configs(self) -> Dict[str, StreamConfig]:
        return {
            'continuous':
                StreamConfig(
                    'continuous', 'continuous', StreamConfig.BIT_VOLTS_AC, StreamConfig.OFFSET_AC
                )
        }

    def num_channels(self) -> int:
        return 32

    def get_amp_data_for_stream(self, stream_name: str, samples: "Samples") -> np.ndarray:
        if stream_name == 'continuous':
            return samples.amp
        raise ValueError(f"Unknown stream for RHD: {stream_name}")


class RHSStreamer(DeviceStreamer):

    def create_stream_configs(self) -> Dict[str, StreamConfig]:
        return {
            'AC': StreamConfig('AC', 'AC', StreamConfig.BIT_VOLTS_AC, StreamConfig.OFFSET_AC),
            'DC': StreamConfig('DC', 'DC', StreamConfig.BIT_VOLTS_DC, StreamConfig.OFFSET_DC)
        }

    def num_channels(self) -> int:
        return 16

    def get_amp_data_for_stream(self, stream_name: str, samples: "Samples") -> np.ndarray:
        if stream_name == 'AC':
            return samples.amp[..., 1]
        elif stream_name == 'DC':
            return samples.amp[..., 0]
        raise ValueError(f"Unknown stream for RHS: {stream_name}")


@dataclass
class StreamWriter:
    stream_config: StreamConfig
    stream_path: pathlib.Path
    streamer: DeviceStreamer
    sample_rate: float

    sample_count: int = 0
    n_channels: Optional[int] = None

    _files: Dict[str, Dict] = field(default_factory=dict)

    def __post_init__(self):
        self.stream_path.mkdir(parents=True, exist_ok=True)
        self._files = {
            'sample_numbers':
                {
                    'path': self.stream_path / "sample_numbers.npy",
                    'dtype': np.dtype(np.int64),
                    'handle': None,
                    'is_npy': True,
                },
            'timestamps':
                {
                    'path': self.stream_path / "timestamps.npy",
                    'dtype': np.dtype(np.float64),
                    'handle': None,
                    'is_npy': True,
                },
            'data':
                {
                    'path': self.stream_path / "continuous.dat",
                    'dtype': np.dtype(np.int16),
                    'handle': None,
                    'is_npy': False,
                },
        }

    @staticmethod
    def _get_npy_header(dtype: np.dtype, shape: Optional[tuple] = None) -> bytes:
        if shape is None:
            shape = ()
        desc = {
            "descr": np.lib.format.dtype_to_descr(dtype),
            "fortran_order": False,
            "shape": shape,
        }
        h = np.lib.format.magic(1, 0)
        h += int(4096 - 10).to_bytes(2, 'little')
        header = str(desc).encode('ASCII')
        h += header + b'\x20' * (4095 - len(header) - len(h))
        h += b'\n'
        assert len(h) % 64 == 0
        return h

    def open(self):
        for data_info in self._files.values():
            data_info['handle'] = open(data_info['path'], 'wb')
            if data_info.get('is_npy'):
                header = self._get_npy_header(data_info['dtype'])
                data_info['handle'].write(header)
        return self.stream_path

    def write_sample_data(self, samples: "Samples"):
        n_samples = samples.n
        if n_samples == 0:
            return

        amp_data = self.streamer.get_amp_data_for_stream(self.stream_config.name, samples)
        reshaped = amp_data.reshape(n_samples, -1)

        result_amp = None
        if self.stream_config.name == 'DC':
            reshaped = reshaped & 0x3FE

        result_amp = reshaped.astype(np.int32) - self.stream_config.offset
        # result_amp = (reshaped.astype(np.int32) -
        #               self.stream_config.offset).astype(np.float64) * self.stream_config.bit_volts

        timestamps = samples.timestamp.astype(np.float64) / 1_000_000.0

        self._files['sample_numbers']['handle'].write(samples.sample_index.tobytes())
        self._files['data']['handle'].write(result_amp.astype(np.int16).tobytes())
        self._files['timestamps']['handle'].write(timestamps.tobytes())
        self.sample_count += n_samples

    def close(self):
        for data_info in self._files.values():
            handle = data_info.get('handle')
            if handle:
                if data_info.get('is_npy'):
                    handle.seek(0)
                    final_shape = (self.sample_count,)
                    header = self._get_npy_header(data_info['dtype'], shape=final_shape)
                    handle.write(header)
                handle.close()
                data_info['handle'] = None
        self._files.clear()
