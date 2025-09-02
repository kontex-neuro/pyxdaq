from pyxdaq.datablock import DataBlock, OldDataBlock, _RHS_HEADER_MAGIC, _RHD_HEADER_MAGIC
import numpy as np


def get_sample_size(rhs: bool, datastreams: int, device_timestamp: bool) -> int:
    if rhs:
        return (
            12  # header (8 bytes magic + 4 bytes sample_index)
            + 3 * datastreams * 2 * 2  # aux (3 channels * datastreams * 2 bytes * 2 DC/AC)
            + 1 * datastreams * 2 * 2  # aux0 (1 channel * datastreams * 2 bytes * 2 DC/AC)
            + 16 * datastreams * 2 * 2  # amp (16 channels * datastreams * 2 bytes * 2 DC/AC)
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


def create_buffer(
    rhs: bool,
    n_samples: int,
    sample_size: int,
) -> bytes:
    magic = _RHS_HEADER_MAGIC if rhs else _RHD_HEADER_MAGIC
    data = np.random.randint(0, 256, (n_samples, sample_size), dtype=np.uint8)
    data[:, :8] = np.frombuffer(magic.to_bytes(8, "little"), dtype=np.uint8).reshape(1, 8)
    return data.tobytes()


if __name__ == "__main__":
    buffer_size = 1 * 1024 * 1024
    rhs_options = [False, True]
    datastreams_options = {False: range(0, 32), True: range(0, 8)}
    device_timestamp_options = [False, True]
    chunk_sizes = [2**i for i in range(14, 26)]

    for rhs in rhs_options:
        for datastreams in datastreams_options[rhs]:
            for device_timestamp in device_timestamp_options:
                sample_size = get_sample_size(rhs, datastreams, device_timestamp)
                n_samples = max(1, buffer_size // sample_size)
                buffer = create_buffer(rhs, n_samples, sample_size)

                for chunk_size in chunk_sizes:
                    samples_per_chunk = chunk_size // sample_size
                    chunk_bytes = samples_per_chunk * sample_size

                    for i in range(0, len(buffer), chunk_bytes):
                        old_block = OldDataBlock.from_buffer(
                            rhs, sample_size, buffer[i:i + chunk_bytes], datastreams,
                            device_timestamp
                        )
                        new_block = DataBlock.from_buffer(
                            rhs, sample_size, buffer[i:i + chunk_bytes], datastreams,
                            device_timestamp
                        )

                        if old_block.to_samples() != new_block.to_samples():
                            print(
                                f"Mismatch -> rhs={rhs}, datastreams={datastreams}, "
                                f"timestamp={device_timestamp}, chunk_offset={i}"
                            )
