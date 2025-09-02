import struct
from dataclasses import dataclass
from typing import Union

import numpy as np

_uint16le = np.dtype("u2").newbyteorder("<")
_uint32le = np.dtype("u4").newbyteorder("<")
_RHD_HEADER_MAGIC = 0xD7A22AAA38132A53
_RHS_HEADER_MAGIC = 0x8D542C8A49712F0B


@dataclass
class Sample:
    """
    Represents a single sample at time `sample_index` from the XDAQ data stream.
    Headstages can vary in data streams and channels.

    Attributes:
        sample_index: Acquisition sample count / timestep / timestamp
            Type: 32-bit unsigned integer
            Shape: scalar
        aux: Auxiliary data channels from the headstage
            Type: 16-bit unsigned integer
            Shape: [3, datastreams] for RHD; [4, datastreams, 2] for RHS
        amp: Headstage amplifier channels
            Type: 16-bit unsigned integer
            Shape: [32, datastreams] for RHD; [16, datastreams, 2] for RHS
        timestamp: XDAQ device timestamp in microseconds (Gen 2 Only)
            Type: 64-bit unsigned integer or None
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

    sample_index: Union[int, np.ndarray]
    aux: np.ndarray
    amp: np.ndarray
    timestamp: Union[None, int, np.ndarray]
    adc: np.ndarray
    ttlin: np.ndarray
    ttlout: np.ndarray
    dac: Union[None, np.ndarray]
    stim: Union[None, np.ndarray]


@dataclass
class Samples(Sample):
    """
    Collection of samples; first dimension is sample index.

    Attributes:
        sample_index: Acquisition sample counts / timesteps / timestamps
            Type: 32-bit unsigned integer
            Shape: [n_samples]
        aux: Auxiliary data channels from the headstage
            Type: 16-bit unsigned integer
            Shape: [n_samples, 3, datastreams] for RHD; [n_samples, 4, datastreams, 2] for RHS
        amp: Headstage amplifier channels
            | Dimension   | Range | Range | Description                                                               |
            | Headstage   |  RHD  |  RHS  |                                                                           |
            |-------------|-------|-------|---------------------------------------------------------------------------|
            | n_samples   | N     | N     | Number of samples                                                         |
            | channels    | 32    | 16    | Number of channels per datastream                                         |
            | datastreams | S     | S     | Number of datastreams: depends on type and numbers of attached headstages |
            | [DC, AC]    | None  | 2     | DC/AC amplifier channel (RHS only); 0: DC low-gain, 1: AC high-gain       |
            Type: 16-bit unsigned integer
            Shape: [n_samples, channels, datastreams]           for RHD;
                   [n_samples, channels, datastreams, [DC, AC]] for RHS
        timestamp: XDAQ device timestamp in microseconds (Gen 2 Only)
            Type: 64-bit unsigned integer or None
            Shape: [n_samples] or None
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
    samples: np.ndarray

    @classmethod
    def from_buffer(
        cls,
        rhs: bool,
        sample_size: int,
        buffer: Union[bytes, bytearray, memoryview],
        datastreams: int,
        device_timestamp: bool,
    ) -> "DataBlock":
        magic = struct.unpack("<Q", buffer[:8])[0]
        if magic != (_RHS_HEADER_MAGIC if rhs else _RHD_HEADER_MAGIC):
            raise ValueError(f"Invalid magic number: {magic:016X}")

        fields = []
        fields.append(("magic", "<u8"))
        fields.append(("sample_index", "<u4"))
        if rhs:
            fields.append(("aux", "<u2", (3, datastreams, 2)))
            fields.append(("amp", "<u2", (16, datastreams, 2)))
            fields.append(("aux0", "<u2", (1, datastreams, 2)))
            fields.append(("stim", "<u2", (4, datastreams)))
            fields.append(("pad", "V4"))
        else:
            fields.append(("aux", "<u2", (3, datastreams)))
            fields.append(("amp", "<u2", (32, datastreams)))
            fields.append(("pad", f"V{2 * ((datastreams + 2) % 4)}"))
        if device_timestamp:
            fields.append(("timestamp", "<u8"))
        if rhs:
            fields.append(("dac", "<u2", (8,)))
        fields.append(("adc", "<u2", (8,)))
        fields.append(("ttlin", "<u4", (1,)))
        fields.append(("ttlout", "<u4", (1,)))

        dtype = np.dtype(fields, align=False)
        if sample_size != dtype.itemsize:
            raise ValueError(f"Expected sample_size = {sample_size}, got = {dtype.itemsize}")

        n_samples = len(buffer) // sample_size
        samples = np.frombuffer(buffer, dtype=dtype, count=n_samples)

        return cls(samples=samples)

    def to_samples(self) -> Samples:
        s = self.samples

        aux = np.concatenate([s["aux0"], s["aux"]], axis=1) if "aux0" in s.dtype.names else s["aux"]

        return Samples(
            np.ascontiguousarray(s["sample_index"]),
            np.ascontiguousarray(aux),
            np.ascontiguousarray(s["amp"]),
            np.ascontiguousarray(s["timestamp"]) if "timestamp" in s.dtype.names else None,
            np.ascontiguousarray(s["adc"]),
            np.ascontiguousarray(s["ttlin"]),
            np.ascontiguousarray(s["ttlout"]),
            np.ascontiguousarray(s["dac"]) if "dac" in s.dtype.names else None,
            np.ascontiguousarray(s["stim"]) if "stim" in s.dtype.names else None,
            len(s),
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


def get_sample_size(rhs: bool, datastreams: int, device_timestamp: bool) -> int:
    if rhs:
        return (
            12  # header (8 bytes magic + 4 bytes sample_index)
            + 3 * datastreams * 4  # aux (3 channels * datastreams * 4 bytes)
            + 1 * datastreams * 4  # aux0 (1 channel * datastreams * 4 bytes)
            + 16 * datastreams * 4  # amp (16 channels * datastreams * 4 bytes)
            + 4 * datastreams * 2  # stim (4 channels * datastreams * 2 bytes)
            + 4  # padding
            + 8 * (1 if device_timestamp else 0)  # timestamp
            + 16  # dac
            + 16  # adc
            + 4  # ttlin
            + 4  # ttlout
        )
    else:
        return (
            12  # header (8 bytes magic + 4 bytes sample_index)
            + 3 * datastreams * 2  # aux (3 channels * datastreams * 2 bytes)
            + 32 * datastreams * 2  # amp (32 channels * datastreams * 2 bytes)
            + 2 * ((datastreams + 2) % 4)  # padding
            + 8 * (1 if device_timestamp else 0)  # timestamp
            + 16  # adc
            + 4  # ttlin
            + 4  # ttlout
        )
