import struct
from dataclasses import dataclass
from typing import List, Union

import numpy as np

_uint16le = np.dtype('u2').newbyteorder('<')
_uint32le = np.dtype('u4').newbyteorder('<')
_RHD_HEADER_MAGIC = 0xD7A22AAA38132A53
_RHS_HEADER_MAGIC = 0x8D542C8A49712F0B


@dataclass
class Sample:
    """
    Represents a single sample at time `ts` from the XDAQ data stream.
    """
    ts: int
    aux: np.ndarray
    amp: np.ndarray
    adc: np.ndarray
    ttlin: np.ndarray
    ttlout: np.ndarray
    dac: Union[None, np.ndarray]
    stim: Union[None, np.ndarray]

    @classmethod
    def from_buffer(
        cls, rhs: bool, buffer: Union[bytearray, memoryview], datastreams: int
    ) -> 'Sample':
        """
        Deserialize a single sample from a buffer.
        Keeps the same memory layout as the original data for further optimization.
        """
        idx = 0
        magic, ts = struct.unpack("<QI", buffer[idx:12])
        if magic != (_RHS_HEADER_MAGIC if rhs else _RHD_HEADER_MAGIC):
            raise ValueError(f"Invalid magic: {magic:016X}")
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
                buffer[idx:], dtype=_uint16le, count=1 * datastreams * 2
            ).reshape((1, datastreams, 2))
            aux = np.concatenate((aux0, aux), axis=0)
            idx += 1 * datastreams * 2 * 2

            stim = np.frombuffer(
                buffer[idx:], dtype=_uint16le, count=4 * datastreams
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
    A collection of samples, the first dimension represents the sample index.
    """
    n: int

    def device_name(self):
        if self.n != 128:
            raise ValueError("Unable to determine device name for non-128 sample data block")
        if self.stim is None:
            return self.aux[[32, 33, 34, 35, 36, 24, 25, 26], 2, :]
        else:
            rom = self.aux[:, 0, :, :][58:61, :, 0]
            aux = np.array(rom).view(np.uint8).reshape(
                (rom.shape[0], rom.shape[1], 2)
            ).transpose(1, 0, 2).reshape((rom.shape[1], -1))
            return aux[:, :0:-1].T

    def device_id(self):
        if self.n != 128:
            raise ValueError("Unable to determine device ID for non-128 sample data block")
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
    Raw data block which keeps the original memory layout.
    """
    samples: List[Sample]
    # Samples x C x Datastreams : [C x Datastreams] [C x Datastreams] ... [C x Datastreams]

    @classmethod
    def from_buffer(
        cls, rhs, sample_size, buffer: Union[bytearray, memoryview], datastreams: int
    ) -> 'DataBlock':
        return cls(
            [
                Sample.from_buffer(rhs, buffer[i:i + sample_size], datastreams)
                for i in range(0, len(buffer), sample_size)
            ]
        )

    def to_samples(self) -> Samples:
        """
        Concatenate all samples into a single Samples object.
        This method breaks the original memory layout.
        """
        return Samples(
            np.array([s.ts for s in self.samples]), np.stack([s.aux for s in self.samples]),
            np.stack([s.amp for s in self.samples]), np.stack([s.adc for s in self.samples]),
            np.stack([s.ttlin for s in self.samples]), np.stack([s.ttlout for s in self.samples]),
            None if self.samples[0].dac is None else np.stack([s.dac for s in self.samples]),
            None if self.samples[0].dac is None else np.stack([s.stim for s in self.samples]),
            len(self.samples)
        )


def amplifier2mv(amp: np.array):
    return (amp.astype(np.float32) - 32768) * 0.195


def adc2v(adc: np.array):
    return (adc.astype(np.float32) - 32768) * 0.0003125
