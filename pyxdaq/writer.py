import os
import json
from datetime import datetime
from .xdaq import XDAQ
from .datablock import Samples

import numpy as np

BIT_VOLTS_AC = 0.195  # uV per bit for AC channels
BIT_VOLTS_DC = -19.23  # uV per bit for DC channels (-19.23 mV)

OFFSET_AC = 32768
OFFSET_DC = 512


class OpenEphysWriter:
    SOURCE_PROCESSOR_NAME = "Acquisition Board"
    SOURCE_PROCESSOR_ID = 100

    def __init__(
        self,
        xdaq: "XDAQ",
        root_path: str,
        record_node: str = "Record Node 101",
        gui_version: str = "0.6.4"
    ):
        # 1. Create the top-level session directory
        self.xdaq = xdaq
        self.gui_version = gui_version
        session_ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.session_path = os.path.join(root_path, session_ts)

        # The analysis library expects a "Record Node" subdirectory.
        self.record_node_path = os.path.join(self.session_path, record_node)
        os.makedirs(self.record_node_path, exist_ok=True)

        self.experiment_index = 0
        self.recording_index = 0
        self._is_recording = False

        # File handling attributes
        self._file_handles = {}
        self._memmaps = {}
        self._memmap_file_handles = {}
        self._memmap_paths = {}
        self._sample_counts = {}
        self._memmap_capacities = {}
        self.memmap_chunk_size_samples = 0
        self.sample_rate = 0
        self.experiment_path = None
        self.recording_path = None
        self.stream_paths = {}
        self.stream_names = {}

    def __enter__(self):
        self.start_recording()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_recording()

    def _write_structure_oebin(self):
        """Creates the structure.oebin JSON file for all streams in the recording."""
        continuous_streams = []

        for key, stream_display_name in self.stream_names.items():
            num_channels = self.xdaq.numDataStream * (16 if self.xdaq.rhs else 32)
            bit_volts = BIT_VOLTS_AC if key == 'AC' or not self.xdaq.rhs else BIT_VOLTS_DC
            folder_name = f"{self.SOURCE_PROCESSOR_NAME}-{self.SOURCE_PROCESSOR_ID}.{stream_display_name}"

            stream_info = {
                "stream_name":
                    stream_display_name,
                "folder_name":
                    folder_name,
                "sample_rate":
                    float(self.xdaq.sampleRate.rate),
                "num_channels":
                    num_channels,
                "source_processor_id":
                    self.SOURCE_PROCESSOR_ID,
                "source_processor_name":
                    self.SOURCE_PROCESSOR_NAME,
                "channels":
                    [
                        {
                            "channel_name": f"CH{i+1}",
                            "bit_volts": bit_volts
                        } for i in range(num_channels)
                    ]
            }
            continuous_streams.append(stream_info)

        structure = {
            "GUI version": self.gui_version,
            "continuous": continuous_streams,
            "events": [],
            "spikes": []
        }
        oebin_path = os.path.join(self.recording_path, "structure.oebin")
        with open(oebin_path, 'w') as f:
            json.dump(structure, f, indent=4)

    def start_new_experiment(self):
        self.experiment_index += 1
        self.recording_index = 0
        self.experiment_path = os.path.join(
            self.record_node_path, f"experiment{self.experiment_index}"
        )
        os.makedirs(self.experiment_path, exist_ok=True)

    def start_recording(self):
        if not self.experiment_path:
            self.start_new_experiment()

        self.recording_index += 1
        self.recording_path = os.path.join(self.experiment_path, f"recording{self.recording_index}")
        os.makedirs(self.recording_path, exist_ok=True)

        stream_keys = ['AC', 'DC'] if self.xdaq.rhs else ['continuous']
        self.stream_names = {'AC': 'AC', 'DC': 'DC', 'continuous': 'Rhythm_Data'}
        self.stream_names = {key: self.stream_names[key] for key in stream_keys}

        self.sample_rate = self.xdaq.sampleRate.rate
        self.memmap_chunk_size_samples = self.sample_rate * 60

        for key in stream_keys:
            stream_display_name = self.stream_names[key]
            folder_name = f"{self.SOURCE_PROCESSOR_NAME}-{self.SOURCE_PROCESSOR_ID}.{stream_display_name}"
            stream_path = os.path.join(self.recording_path, "continuous", folder_name)
            self.stream_paths[key] = stream_path
            os.makedirs(stream_path, exist_ok=True)

            self._file_handles[key] = open(os.path.join(stream_path, "continuous.dat"), 'ab')
            self._memmap_capacities[key] = self.memmap_chunk_size_samples
            self._sample_counts[key] = 0

            self._memmap_paths[key] = {
                'sample_numbers': os.path.join(stream_path, "sample_numbers.mmap"),
                'timestamps': os.path.join(stream_path, "timestamps.mmap"),
            }
            self._memmap_file_handles[key] = {}
            self._memmaps[key] = {}

            sn_fp = open(self._memmap_paths[key]['sample_numbers'], 'w+b')
            sn_fp.truncate(self._memmap_capacities[key] * np.dtype(np.int64).itemsize)
            self._memmap_file_handles[key]['sample_numbers'] = sn_fp
            self._memmaps[key]['sample_numbers'] = np.memmap(
                sn_fp, dtype=np.int64, mode='r+', shape=(self._memmap_capacities[key],)
            )

            ts_fp = open(self._memmap_paths[key]['timestamps'], 'w+b')
            ts_fp.truncate(self._memmap_capacities[key] * np.dtype(np.float64).itemsize)
            self._memmap_file_handles[key]['timestamps'] = ts_fp
            self._memmaps[key]['timestamps'] = np.memmap(
                ts_fp, dtype=np.float64, mode='r+', shape=(self._memmap_capacities[key],)
            )
            print(f"Started recording in: {stream_path}")

        self._write_structure_oebin()
        self._is_recording = True

    def _resize_memmaps(self, key: str, required_capacity: int):
        new_capacity = self._memmap_capacities[key]
        while new_capacity < required_capacity:
            new_capacity += self.memmap_chunk_size_samples

        for mmap_key in self._memmaps[key]:
            self._memmaps[key][mmap_key].flush()
        del self._memmaps[key]

        fp_sn = self._memmap_file_handles[key]['sample_numbers']
        fp_ts = self._memmap_file_handles[key]['timestamps']
        fp_sn.truncate(new_capacity * np.dtype(np.int64).itemsize)
        fp_ts.truncate(new_capacity * np.dtype(np.float64).itemsize)

        self._memmaps[key] = {
            'sample_numbers': np.memmap(fp_sn, dtype=np.int64, mode='r+', shape=(new_capacity,)),
            'timestamps': np.memmap(fp_ts, dtype=np.float64, mode='r+', shape=(new_capacity,))
        }
        self._memmap_capacities[key] = new_capacity
        print(
            f"\nResized memmap files for stream '{key}' to capacity for "
            f"{new_capacity // self.sample_rate} seconds."
        )

    def stop_recording(self):
        if not self._is_recording:
            return

        for key, stream_path in self.stream_paths.items():
            try:
                self._file_handles[key].close()
                for mmap_key in self._memmaps[key]:
                    self._memmaps[key][mmap_key].flush()

                final_sample_count = self._sample_counts[key]

                sn_mmap = np.memmap(
                    self._memmap_paths[key]['sample_numbers'],
                    dtype=np.int64,
                    mode='r',
                    shape=(final_sample_count,)
                )
                np.save(os.path.join(stream_path, "sample_numbers.npy"), sn_mmap)
                del sn_mmap
                os.remove(self._memmap_paths[key]['sample_numbers'])

                ts_mmap = np.memmap(
                    self._memmap_paths[key]['timestamps'],
                    dtype=np.float64,
                    mode='r',
                    shape=(final_sample_count,)
                )
                np.save(os.path.join(stream_path, "timestamps.npy"), ts_mmap)
                del ts_mmap
                os.remove(self._memmap_paths[key]['timestamps'])

                print(f"Stopped recording. Data saved in: {stream_path}")

            finally:
                for fp_key in self._memmap_file_handles[key]:
                    self._memmap_file_handles[key][fp_key].close()

        self._is_recording = False

    def write_data(self, samples: "Samples"):
        if not self._is_recording:
            return
        n_samples = samples.n
        if n_samples == 0:
            return

        for key in self.stream_paths.keys():
            required_capacity = self._sample_counts[key] + n_samples
            if required_capacity > self._memmap_capacities[key]:
                self._resize_memmaps(key, required_capacity)

            offset = self._sample_counts[key]
            self._memmaps[key]['sample_numbers'][offset:offset + n_samples] = samples.sample_index
            self._memmaps[key]['timestamps'][offset:offset + n_samples] = samples.timestamp.astype(
                np.float64
            ) / 1_000_000.0
            self._sample_counts[key] += n_samples

        if self.xdaq.rhs:
            amp_data_ac = samples.amp[..., 1]
            reshaped_ac = amp_data_ac.reshape(n_samples, -1)
            ac_int16 = (reshaped_ac.astype(np.int32) - OFFSET_AC).astype(np.int16)
            self._file_handles['AC'].write(ac_int16.tobytes())

            amp_data_dc = samples.amp[..., 0]
            reshaped_dc = amp_data_dc.reshape(n_samples, -1)
            dc_10bit = reshaped_dc & 0x3FE
            dc_int16 = (dc_10bit.astype(np.int32) - OFFSET_DC).astype(np.int16)
            self._file_handles['DC'].write(dc_int16.tobytes())
        else:
            amp_data = samples.amp
            reshaped_amp = amp_data.reshape(n_samples, -1)
            amp_data_int16 = (reshaped_amp.astype(np.int32) - OFFSET_AC).astype(np.int16)
            self._file_handles['continuous'].write(amp_data_int16.tobytes())
