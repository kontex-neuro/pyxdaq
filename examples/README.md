# Examples

| Name                             | Description                                                                 |
|---|---|
| `plating.py`                     | Facilitates a looped sequence of stimulation and impedance measurements. |
| `plot_stim.py`                   | Generates a stimulation sequence, executes stimulation, and visualizes outcomes. |
| `realtime_process.py`            | Realtime data acquisition and processing. |
| `realtime_stimulation.py`        | Realtime data acquisition with stimulation triggering based on channel amplitude thresholds. |
| `realtime_power_threshold.py`    | Self-contained closed loop: live RHS signal → band power → threshold → TTL output, plus the two detector classes. Start here. |
| `pipeline_latency_detector.py`   | Measures that same loop on your hardware — software round-trip per stage, plus the detector's own response time. Imports the detectors from the script above. |
| `latency_analyze.py`             | Per-stage breakdown of a single detector run: transport, parsing and compute. |
| `run_impedance_measurements.py`  | Impedance measurement across various frequencies for performance analysis |
| `impedance_test_analysis.py`     | Analysis of impedance measurement data from `run_impedance_measurements.py`. |