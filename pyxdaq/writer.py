import os
import json
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

BIT_VOLTS = 0.195
UINT16_TO_INT16_OFFSET = 32768

if TYPE_CHECKING:
    from .xdaq import XDAQ
    from .datablock import Samples


class OpenEphysWriter:

    def __init__(self, xdaq: "XDAQ", root_path: str, record_node: str = "Record Node 101"):
        # 1. Create the top-level session directory
        self.xdaq = xdaq
        session_ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.session_path = os.path.join(root_path, session_ts)
        self.record_node_path = os.path.join(self.session_path, record_node)

        os.makedirs(self.record_node_path, exist_ok=True)

        self.experiment_index = 0
        self.recording_index = 0
        self._is_recording = False

        # File handling for continuous data and memory-mapped timestamp/sample_number data
        self._file_handles = {}
        self._memmaps = {}
        self._memmap_file_handles = {}
        self._memmap_paths = {}
        self._sample_count = 0
        self._memmap_capacity = 0
        self.memmap_chunk_size_samples = 0
        self.sample_rate = 0  # To be set in start_recording

        self.recording_path = None

    def __enter__(self):
        self.start_recording()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_recording()

    def _write_structure_oebin(self, stream_name: str):
        """Creates the structure.oebin JSON file."""

        # For now, we assume a single continuous stream from the amplifier
        num_channels = self.xdaq.numDataStream * (16 if self.xdaq.rhs else 32)

        structure = {
            "GUI version": "0.6.4",
            "continuous":
                [
                    {
                        "stream_name":
                            stream_name,
                        "folder_name":
                            stream_name,
                        "sample_rate":
                            self.xdaq.sampleRate.rate,
                        "num_channels":
                            num_channels,
                        "source_processor_id":
                            100,
                        "source_processor_name":
                            "Acquisition Board",
                        "channels":
                            [
                                {
                                    "channel_name": f"CH{i+1}",
                                    "bit_volts": BIT_VOLTS  # From amplifier2uv
                                } for i in range(num_channels)
                            ]
                    }
                ],
            "events": [],
            "spikes": []
        }

        # The file goes in the 'recording' directory
        oebin_path = os.path.join(self.recording_path, "structure.oebin")
        with open(oebin_path, 'w') as f:
            json.dump(structure, f, indent=4)

    def start_new_experiment(self):
        self.experiment_index += 1
        self.recording_index = 0  # Reset recording index for new experiment
        self.experiment_path = os.path.join(
            self.record_node_path, f"experiment{self.experiment_index}"
        )
        os.makedirs(self.experiment_path, exist_ok=True)

    def start_recording(self, *, stream_name: str = "XDAQ-100.Rhythm_Data"):
        if not hasattr(self, 'experiment_path'):
            self.start_new_experiment()

        self.recording_index += 1
        self.recording_path = os.path.join(self.experiment_path, f"recording{self.recording_index}")

        # 2. Create the stream directory for continuous data
        self.stream_path = os.path.join(self.recording_path, "continuous", stream_name)
        os.makedirs(self.stream_path, exist_ok=True)

        # Write the structure.oebin file
        self._write_structure_oebin(stream_name)

        # 3. Open 'continuous.dat' for binary appending. This is efficient as
        # the OS handles buffering writes to disk.
        self._file_handles['continuous'] = open(
            os.path.join(self.stream_path, "continuous.dat"), 'ab'
        )

        # 4. Initialize memory-mapped files for high-throughput writing of
        #    timestamps and sample numbers, avoiding high RAM usage.
        # Pre-allocate for 1 minute of data, and resize by the same amount.
        self.sample_rate = self.xdaq.sampleRate.rate
        self.memmap_chunk_size_samples = self.sample_rate * 60
        self._memmap_capacity = self.memmap_chunk_size_samples
        self._sample_count = 0

        self._memmap_paths = {
            'sample_numbers': os.path.join(self.stream_path, "sample_numbers.mmap"),
            'timestamps': os.path.join(self.stream_path, "timestamps.mmap"),
        }

        # Create and size the raw files for memory mapping
        sn_fp = open(self._memmap_paths['sample_numbers'], 'w+b')
        sn_fp.truncate(self._memmap_capacity * np.dtype(np.int64).itemsize)
        self._memmap_file_handles['sample_numbers'] = sn_fp
        self._memmaps['sample_numbers'] = np.memmap(
            sn_fp, dtype=np.int64, mode='r+', shape=(self._memmap_capacity,)
        )

        ts_fp = open(self._memmap_paths['timestamps'], 'w+b')
        ts_fp.truncate(self._memmap_capacity * np.dtype(np.float64).itemsize)
        self._memmap_file_handles['timestamps'] = ts_fp
        self._memmaps['timestamps'] = np.memmap(
            ts_fp, dtype=np.float64, mode='r+', shape=(self._memmap_capacity,)
        )

        self._is_recording = True
        print(f"Started recording in: {self.stream_path}")

    def _resize_memmaps(self, required_capacity: int):
        """Resizes the memory-mapped files to a new capacity."""
        new_capacity = self._memmap_capacity
        while new_capacity < required_capacity:
            new_capacity += self.memmap_chunk_size_samples

        # Flush, close, and release the existing memory maps
        for key in list(self._memmaps.keys()):
            self._memmaps[key].flush()
            del self._memmaps[key]

        # Resize the underlying files
        for key, fp in self._memmap_file_handles.items():
            dtype = np.int64 if key == 'sample_numbers' else np.float64
            fp.truncate(new_capacity * np.dtype(dtype).itemsize)

        # Re-establish the memory maps with the new capacity
        self._memmaps['sample_numbers'] = np.memmap(
            self._memmap_file_handles['sample_numbers'],
            dtype=np.int64,
            mode='r+',
            shape=(new_capacity,)
        )
        self._memmaps['timestamps'] = np.memmap(
            self._memmap_file_handles['timestamps'],
            dtype=np.float64,
            mode='r+',
            shape=(new_capacity,)
        )
        self._memmap_capacity = new_capacity
        print(f"\nResized memmap files to capacity for {new_capacity // self.sample_rate} seconds.")

    def stop_recording(self):
        if not self._is_recording:
            return

        # Close the continuous data file
        self._file_handles['continuous'].close()

        # Finalize memory-mapped files: flush, convert to .npy, and clean up.
        # Flush any remaining data to disk
        for key in self._memmaps:
            self._memmaps[key].flush()

        # Close file handles for the memory-mapped files
        for fp in self._memmap_file_handles.values():
            fp.close()

        # Create final .npy files from the memory-mapped temp files
        final_sample_count = self._sample_count

        # Sample Numbers
        sn_mmap = np.memmap(
            self._memmap_paths['sample_numbers'],
            dtype=np.int64,
            mode='r',
            shape=(final_sample_count,)
        )
        sn_path = os.path.join(self.stream_path, "sample_numbers.npy")
        np.save(sn_path, sn_mmap)
        del sn_mmap  # Release memmap
        os.remove(self._memmap_paths['sample_numbers'])  # Clean up temp file

        # Timestamps
        ts_mmap = np.memmap(
            self._memmap_paths['timestamps'],
            dtype=np.float64,
            mode='r',
            shape=(final_sample_count,)
        )
        ts_path = os.path.join(self.stream_path, "timestamps.npy")
        np.save(ts_path, ts_mmap)
        del ts_mmap  # Release memmap
        os.remove(self._memmap_paths['timestamps'])  # Clean up temp file

        self._is_recording = False
        print(f"Stopped recording. Data saved in: {self.stream_path}")

    def write_data(self, samples: "Samples"):  # Expects a 'Samples' object
        if not self._is_recording:
            return

        n_samples = samples.n
        if n_samples == 0:
            return

        # --- Data Transformation and Saving ---

        # 7. Write sample_index and timestamp to memory-mapped files
        required_capacity = self._sample_count + n_samples
        if required_capacity > self._memmap_capacity:
            self._resize_memmaps(required_capacity)

        offset = self._sample_count
        self._memmaps['sample_numbers'][offset:offset + n_samples] = samples.sample_index
        self._memmaps['timestamps'][offset:offset + n_samples] = samples.timestamp.astype(
            np.float64
        ) / 1_000_000.0
        self._sample_count += n_samples

        # 8. Process and write amplifier data
        amp_data = samples.amp

        # The Samples object from the new API provides data with channels already
        # ordered correctly for reshaping.
        # Original transpose operations are no longer needed.

        # Reshape to 2D and convert uint16 to int16
        reshaped_amp = amp_data.reshape(n_samples, -1)
        amp_data_int16 = (reshaped_amp.astype(np.int32) - UINT16_TO_INT16_OFFSET).astype(np.int16)

        # Data is now interleaved by sample, so we can write directly
        self._file_handles['continuous'].write(amp_data_int16.tobytes())
