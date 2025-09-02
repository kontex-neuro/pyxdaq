from pyxdaq.datablock import DataBlock, _RHS_HEADER_MAGIC, _RHD_HEADER_MAGIC
import timeit
import numpy as np
import itertools


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


def benchmark_from_buffer(configs):
    buffer_size = 1 * 1024 * 1024

    print("Mode\tDatastreams\tSample Size\tChunk Size\tTime\t\t\tThroughput (MB/s)")

    current_datastream = None
    for rhs, datastreams, chunk_size in configs:
        if current_datastream is not None and current_datastream != datastreams:
            print("-" * 100)
        current_datastream = datastreams

        sample_size = get_sample_size(rhs, datastreams, True)
        n_samples = buffer_size // sample_size
        buffer = create_buffer(rhs, n_samples, sample_size)
        samples_per_chunk = chunk_size // sample_size
        chunk_bytes = samples_per_chunk * sample_size

        def benchmark():
            if chunk_size == 0:
                block = DataBlock.from_buffer(
                    rhs,
                    sample_size,
                    buffer,
                    datastreams,
                    True,
                )
                block.to_samples()
            else:
                for i in range(0, len(buffer), chunk_bytes):
                    block = DataBlock.from_buffer(
                        rhs,
                        sample_size,
                        buffer[i:i + chunk_bytes],
                        datastreams,
                        True,
                    )
                    block.to_samples()

        time_taken = timeit.timeit(benchmark, number=10)
        throughput = len(buffer) / 1024**2 / time_taken

        mode = "RHS" if rhs else "RHD"
        print(
            f"{mode}\t{datastreams}\t\t{sample_size}\t\t{chunk_bytes}\t\t{time_taken:f} s\t\t{throughput:.2f} MB/s"
        )


if __name__ == "__main__":
    rhs_options = [False, True]
    datastreams_options = {False: range(0, 32, 4), True: range(0, 8)}
    chunk_sizes = [2**i for i in range(14, 26)]
    configs = []

    for rhs in rhs_options:
        for datastreams in datastreams_options[rhs]:
            configs.extend(itertools.product([rhs], [datastreams], chunk_sizes))

    benchmark_from_buffer(configs)
