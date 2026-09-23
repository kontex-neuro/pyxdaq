import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from .datablock import Samples


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
    """
    Flattens the per-datastream amplifier data of a Samples block into channel columns.

    channels_per_stream, when given, lists how many leading channels of each datastream
    hold real data, in datablock order (see XDAQ.enabled_stream_channels). It exists for
    the RHD2216, which occupies 32 amplifier words per datastream but only fills the
    first 16; the trailing 16 are dummy data and are dropped here rather than written to
    disk. Leaving it None keeps every wire channel.
    """

    CHANNELS_PER_STREAM_ON_WIRE: int

    def __init__(self, channels_per_stream: Optional[Sequence[int]] = None):
        self.channels_per_stream = (
            None if channels_per_stream is None else [int(c) for c in channels_per_stream]
        )
        if self.channels_per_stream is not None:
            for count in self.channels_per_stream:
                if not 0 <= count <= self.CHANNELS_PER_STREAM_ON_WIRE:
                    raise ValueError(
                        f"channels_per_stream entries must be between 0 and "
                        f"{self.CHANNELS_PER_STREAM_ON_WIRE}, got {count}"
                    )

    @abstractmethod
    def create_stream_configs(self) -> Dict[str, StreamConfig]:
        """Creates stream configurations for the device."""

    @abstractmethod
    def get_amp_data_for_stream(self, stream_name: str, samples: "Samples") -> np.ndarray:
        """
        Extracts amplifier data for a specific stream from a Samples object,
        shaped [n_samples, num_channels()] in the order it is written to disk.
        """

    def num_channels(self, num_streams: Optional[int] = None) -> int:
        """Total amplifier channels written per sample across all enabled datastreams."""
        if self.channels_per_stream is not None:
            return sum(self.channels_per_stream)
        if num_streams is None:
            raise ValueError(
                "num_streams is required when the streamer was built without channels_per_stream"
            )
        return num_streams * self.CHANNELS_PER_STREAM_ON_WIRE

    def _flatten(self, amp: np.ndarray) -> np.ndarray:
        # amp: [n_samples, n_streams, CHANNELS_PER_STREAM_ON_WIRE]
        n_samples, n_streams, n_wire = amp.shape
        if self.channels_per_stream is None:
            return amp.reshape(n_samples, -1)
        if len(self.channels_per_stream) != n_streams:
            raise ValueError(
                f"Streamer was configured for {len(self.channels_per_stream)} datastreams "
                f"but received {n_streams}"
            )
        if all(c == n_wire for c in self.channels_per_stream):
            return amp.reshape(n_samples, -1)
        return np.concatenate(
            [amp[:, i, :c] for i, c in enumerate(self.channels_per_stream)], axis=1
        )


class RHDStreamer(DeviceStreamer):
    CHANNELS_PER_STREAM_ON_WIRE = 32

    def create_stream_configs(self) -> Dict[str, StreamConfig]:
        return {
            'continuous':
                StreamConfig(
                    'continuous', 'continuous', StreamConfig.BIT_VOLTS_AC, StreamConfig.OFFSET_AC
                )
        }

    def get_amp_data_for_stream(self, stream_name: str, samples: "Samples") -> np.ndarray:
        if stream_name == 'continuous':
            return self._flatten(samples.amp)
        raise ValueError(f"Unknown stream for RHD: {stream_name}")


class RHSStreamer(DeviceStreamer):
    CHANNELS_PER_STREAM_ON_WIRE = 16

    def create_stream_configs(self) -> Dict[str, StreamConfig]:
        return {
            'AC': StreamConfig('AC', 'AC', StreamConfig.BIT_VOLTS_AC, StreamConfig.OFFSET_AC),
            'DC': StreamConfig('DC', 'DC', StreamConfig.BIT_VOLTS_DC, StreamConfig.OFFSET_DC)
        }

    def get_amp_data_for_stream(self, stream_name: str, samples: "Samples") -> np.ndarray:
        if stream_name == 'AC':
            return self._flatten(samples.amp[..., 1])
        elif stream_name == 'DC':
            return self._flatten(samples.amp[..., 0])
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

        if samples.timestamp is not None:
            timestamps = samples.timestamp.astype(np.float64) / 1_000_000.0
        else:
            timestamps = samples.sample_index.astype(np.float64) / self.sample_rate

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
