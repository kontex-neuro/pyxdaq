# pyxdaq : Python Interface and Tools for XDAQ

## Installation
Follow these instructions to install the library.

### Prerequisites
Ensure you have the following installed:
- [Git](https://git-scm.com/) - For cloning the repository.
- [Python](https://www.python.org/downloads/) - Version 3.9 or higher.

### macOS
Open a Terminal window and run the following commands:

```shell
git clone https://github.com/kontex-neuro/pyxdaq.git
cd pyxdaq
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install .
```

### Windows
Open PowerShell by Shift + Right-Clicking in the folder where you want to install the project, and enter:

```powershell
git clone https://github.com/kontex-neuro/pyxdaq.git
cd pyxdaq
python --version # Ensure Python 3.9 or higher is installed
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install .
```

## Running the self diagnosis script
Running the Self-Diagnosis Script
After installation, it is recommended to run the self-diagnosis script to verify the setup. This script checks:
- If the library is correctly installed.
- If the XDAQ device is detected.
- If the headstage is detected.

Please connect your XDAQ device and headstage before running the script:
```shell
python scripts/self_diagnosis.py
```

## Getting Started
For detailed examples on how to use pyxdaq with your experiments, check out the examples folder.