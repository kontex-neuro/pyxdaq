import struct
from dataclasses import dataclass
from typing import List, Union

import numpy as np

_uint16le = np.dtype("u2").newbyteorder("<")
_uint32le = np.dtype("u4").newbyteorder("<")
_RHD_HEADER_MAGIC = 0xD7A22AAA38132A53
_RHS_HEADER_MAGIC = 0x8D542C8A49712F0B


@dataclass
class Sample:
    """
    Represents a single sample at time `ts` from the XDAQ data stream.
    Headstages can vary in data streams and channels.

    Attributes:
        ts: Acquisition sample count / timestep / timestamp
            Type: 32-bit unsigned integer
            Shape: scalar
        aux: Auxiliary data channels from the headstage
            Type: 16-bit unsigned integer
            Shape: [3, datastreams] for RHD; [4, datastreams, 2] for RHS
        amp: Headstage amplifier channels
            Type: 16-bit unsigned integer
            Shape: [32, datastreams] for RHD; [16, datastreams, 2] for RHS
        adc: Analog input channels
            Type: 16-bit unsigned integer
            Shape: [8]
        ttlin: Digital input channels
            Type: 32-bit unsigned integer
            Shape: [1]
        ttlout: Digital output channels readback
            Type: 32-bit unsigned integer
            Shape: [1]
        dac: Analog output channels readback (RHS only)
            Type: 16-bit unsigned integer
            Shape: [8] (None for RHD)
        stim: Stimulation status readback (RHS only)
            Type: 16-bit unsigned integer
            Shape: [4, datastreams] (None for RHD)
    """

    ts: Union[int, np.ndarray]
    aux: np.ndarray
    amp: np.ndarray
    adc: np.ndarray
    ttlin: np.ndarray
    ttlout: np.ndarray
    dac: Union[None, np.ndarray]
    stim: Union[None, np.ndarray]

    @classmethod
    def from_buffer(
        cls, rhs: bool, buffer: Union[bytes, bytearray, memoryview], datastreams: int
    ) -> "Sample":
        """
        Deserialize a single sample from a buffer, preserving original memory
        layout for optimization. Assumes buffer is exactly one sample; no
        partial-buffer handling.
        """
        idx = 0
        magic, ts = struct.unpack("<QI", buffer[idx:12])
        if magic != (_RHS_HEADER_MAGIC if rhs else _RHD_HEADER_MAGIC):
            raise ValueError(f"Invalid magic number: {magic:016X}")
        idx += 12

        aux = np.frombuffer(
            buffer[idx:], dtype=_uint16le, count=3 * datastreams * (2 if rhs else 1)
        ).reshape([3, datastreams] + ([2] if rhs else []))
        idx += 3 * datastreams * 2 * (2 if rhs else 1)

        amp = np.frombuffer(
            buffer[idx:],
            dtype=_uint16le,
            count=(16 if rhs else 32) * datastreams * (2 if rhs else 1)
        ).reshape([16 if rhs else 32, datastreams] + ([2] if rhs else []))
        idx += (16 if rhs else 32) * datastreams * 2 * (2 if rhs else 1)

        if rhs:
            aux0 = np.frombuffer(
                buffer[idx:],
                dtype=_uint16le,
                count=1 * datastreams * 2,
            ).reshape((1, datastreams, 2))
            aux = np.concatenate((aux0, aux), axis=0)
            idx += 1 * datastreams * 2 * 2

            stim = np.frombuffer(
                buffer[idx:],
                dtype=_uint16le,
                count=4 * datastreams,
            ).reshape(4, datastreams)
            idx += 4 * datastreams * 2
            idx += 4

            dac = np.frombuffer(buffer[idx:], dtype=_uint16le, count=8)
            idx += 16
        else:
            stim = None
            dac = None
            idx += 2 * ((datastreams + 2) % 4)  # padding

        adc = np.frombuffer(buffer[idx:], dtype=_uint16le, count=8)
        idx += 16

        ttlin = np.frombuffer(buffer[idx:], dtype=_uint32le, count=1)
        idx += 4

        ttlout = np.frombuffer(buffer[idx:], dtype=_uint32le, count=1)
        idx += 4
        return cls(ts, aux, amp, adc, ttlin, ttlout, dac, stim)


@dataclass
class Samples(Sample):
    """
    Collection of samples; first dimension is sample index.

    Attributes:
        ts: Acquisition sample counts / timesteps / timestamps
            Type: 32-bit unsigned integer
            Shape: [n_samples]
        aux: Auxiliary data channels from the headstage
            Type: 16-bit unsigned integer
            Shape: [n_samples, 3, datastreams] for RHD; [n_samples, 4, datastreams, 2] for RHS
        amp: Headstage amplifier channels
            Type: 16-bit unsigned integer
            Shape: [n_samples, 32, datastreams] for RHD; [n_samples, 16, datastreams, 2] for RHS
        adc: Analog input channels
            Type: 16-bit unsigned integer
            Shape: [n_samples, 8]
        ttlin: Digital input channels
            Type: 32-bit unsigned integer
            Shape: [n_samples, 1]
        ttlout: Digital output channels readback
            Type: 32-bit unsigned integer
            Shape: [n_samples, 1]
        dac: Analog output channels readback (RHS only)
            Type: 16-bit unsigned integer
            Shape: [n_samples, 8] (None for RHD)
        stim: Stimulation status readback (RHS only)
            Type: 16-bit unsigned integer
            Shape: [n_samples, 4, datastreams] (None for RHD)
        n: Number of samples in the collection
            Type: int
            Shape: scalar
    """

    n: int

    def device_name(self):
        """
        Extract device name from auxiliary data. Valid only for initialization
        phase and exactly 128 samples; fails otherwise.
        """
        if self.n != 128:
            raise ValueError("Device name extraction requires exactly 128 samples")
        if self.stim is None:
            return self.aux[[32, 33, 34, 35, 36, 24, 25, 26], 2, :]
        else:
            rom = self.aux[:, 0, :, :][58:61, :, 0]
            aux = np.array(rom).view(np.uint8).reshape(
                (rom.shape[0], rom.shape[1], 2)
            ).transpose(1, 0, 2).reshape((rom.shape[1], -1))
            return aux[:, :0:-1].T

    def device_id(self):
        """
        Extract device ID from auxiliary data. Valid only for initialization
        phase and exactly 128 samples; fails otherwise.
        """
        if self.n != 128:
            raise ValueError("Device ID extraction requires exactly 128 samples")
        if self.stim is None:
            return self.aux[19, 2, :], self.aux[23, 2, :]
        else:
            rom = self.aux[:, 0, :, :][56:58, :, 0]
            aux = np.array(rom).view(np.uint8).reshape(
                (rom.shape[0], rom.shape[1], 2)
            ).transpose(1, 0, 2).reshape((rom.shape[1], -1))
            return aux[:, 0].T, np.zeros_like(aux[:, 0].T)


@dataclass
class DataBlock:
    """
    Raw data block preserving original memory layout.
    """

    samples: List[Sample]

    @classmethod
    def from_buffer(
        cls, rhs, sample_size, buffer: Union[bytes, bytearray, memoryview], datastreams: int
    ) -> "DataBlock":
        return cls(
            [
                Sample.from_buffer(rhs, buffer[i:i + sample_size], datastreams)
                for i in range(0, len(buffer), sample_size)
            ]
        )

    def to_samples(self) -> Samples:
        """
        Concatenate samples into a Samples object, discarding original memory layout.
        """
        return Samples(
            np.array([s.ts for s in self.samples]),
            np.stack([s.aux for s in self.samples]),
            np.stack([s.amp for s in self.samples]),
            np.stack([s.adc for s in self.samples]),
            np.stack([s.ttlin for s in self.samples]),
            np.stack([s.ttlout for s in self.samples]),
            None if self.samples[0].dac is None else np.stack([s.dac for s in self.samples]),
            None if self.samples[0].stim is None else np.stack([s.stim for s in self.samples]),
            len(self.samples),
        )


def amplifier2uv(amp: np.ndarray) -> np.ndarray:
    """
    Convert amplifier data to microvolts.
    """
    return (amp.astype(np.float32) - 32768) * 0.195


def adc2v(adc: np.ndarray) -> np.ndarray:
    """
    Convert ADC data to volts.
    """
    return (adc.astype(np.float32) - 32768) * 0.0003125
