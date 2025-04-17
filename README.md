# pyxdaq: Python Interface and Tools for XDAQ

## Installation

### Requirements
- **Python** ≥ 3.11  
- **XDAQ** drivers installed and your device/headstage connected

### From PyPI
```shell
python3 -m pip install --upgrade pip
pip install pyxdaq
```

### From Source
```shell
git clone https://github.com/kontex-neuro/pyxdaq.git
cd pyxdaq
python3 -m venv .venv
source .venv/bin/activate     # on Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install .
```

## Running the Self‑Diagnosis

A built‑in CLI script verifies:
- package installation  
- XDAQ dynamic libraries  
- device enumeration  
- headstage detection  

Run:
```shell
xdaq-diagnosis
```

Results are saved to `diagnostic_report.json` in your current directory.

## Quick Start

See the `examples/` folder.

## License

MIT — see [LICENSE](LICENSE)  