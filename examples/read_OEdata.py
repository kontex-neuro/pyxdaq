from open_ephys.analysis import Session
import sys

if len(sys.argv) != 2:
    print("Usage: python open_rhs.py <folder_path>")
    sys.exit(1)

data_folder_str = sys.argv[1]

# Load the session directory
session = Session(data_folder_str)

# 1. Get the first (and only) RecordNode object
record_node = session.recordnodes[0]

# 2. Access the list of recordings within that RecordNode
#    In your case, there will be one recording at index 0.
recording = record_node.recordings[0]

# 3. The 'recording' object contains a list of continuous data streams.
num_streams = len(recording.continuous)

print(f"Found {num_streams} continuous data streams.")

if num_streams == 1:  # RHD
    stream = recording.continuous[0]
    print(f"Found Stream 0: {stream.metadata['stream_name']}")

    print("\nReading 100 samples from stream, starting at sample 20000...")
    samples = stream.get_samples(start_sample_index=20000, end_sample_index=20100)

    print(f"Shape of samples: {samples.shape}")
    print(samples)

elif num_streams == 2:  # RHS
    ac_stream = recording.continuous[0]
    dc_stream = recording.continuous[1]

    print(f"Found Stream 0: {ac_stream.metadata['stream_name']}")
    print(f"Found Stream 1: {dc_stream.metadata['stream_name']}")

    # 4. For the Binary format, you do not use `set_sample_range`.
    #    You directly request the data slice you want with `get_samples`.
    print("\nReading 100 samples from AC stream, starting at sample 20000...")
    ac_samples = ac_stream.get_samples(start_sample_index=20000, end_sample_index=20100)

    print(f"Shape of AC samples: {ac_samples.shape}")
    print(ac_samples)

    print("\nReading 100 samples from DC stream, starting at sample 20000...")
    dc_samples = dc_stream.get_samples(start_sample_index=20000, end_sample_index=20100)

    print(f"Shape of DC samples: {dc_samples.shape}")
    print(dc_samples)

else:
    print(f"Error: Expected 1 (RHD) or 2 (RHS) streams, but found {num_streams}.")
    sys.exit(1)
