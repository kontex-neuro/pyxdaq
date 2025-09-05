from pyxdaq.datablock import DataBlock, _RHS_HEADER_MAGIC, _RHD_HEADER_MAGIC, get_sample_size
import timeit
import numpy as np
import itertools
import json
import argparse
import datetime


def create_buffer(
    rhs: bool,
    n_samples: int,
    sample_size: int,
) -> bytes:
    magic = _RHS_HEADER_MAGIC if rhs else _RHD_HEADER_MAGIC
    data = np.random.randint(0, 256, (n_samples, sample_size), dtype=np.uint8)
    data[:, :8] = np.frombuffer(magic.to_bytes(8, "little"), dtype=np.uint8).reshape(1, 8)
    return data.tobytes()


def benchmark_from_buffer(configs, n_runs=1):
    buffer_size = 1 * 1024 * 1024 * 512
    results = []

    print("Mode\tDatastreams\tSample Size\tChunk Size\tRun\tTime\t\t\tThroughput (MB/s)")

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

        for run_idx in range(n_runs):

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
                f"{mode}\t{datastreams}\t\t{sample_size}\t\t{chunk_bytes}\t\t{run_idx + 1}\t{time_taken:f} s\t\t{throughput:.2f} MB/s"
            )
            results.append(
                {
                    "mode": mode,
                    "datastreams": datastreams,
                    "sample_size": sample_size,
                    "chunk_size": chunk_bytes,
                    "run": run_idx + 1,
                    "time_taken": time_taken,
                    "throughput_mb_s": throughput
                }
            )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark parsing performance.")
    parser.add_argument(
        "-n", "--n_runs", type=int, default=1, help="Number of times to run each configuration."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output JSON file path. If not provided, a timestamped file will be created.",
    )
    args = parser.parse_args()

    rhs_options = [False, True]
    datastreams_options = {False: range(0, 32 + 1, 8), True: range(0, 8 + 1, 2)}
    chunk_sizes = [2**i for i in range(14, 23)]
    configs = []

    for rhs in rhs_options:
        for datastreams in datastreams_options[rhs]:
            configs.extend(itertools.product([rhs], [datastreams], chunk_sizes))

    results = benchmark_from_buffer(configs, n_runs=args.n_runs)

    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"benchmark_results_{timestamp}.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {output_path}")
