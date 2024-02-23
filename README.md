# pyxdaq : Python Interface and Tools for XDAQ

## Installation
### Prerequisites
* [Git](https://git-scm.com/)
* [Python](https://www.python.org/downloads/) 3.9 or higher

### macOS
```shell
git clone https://github.com/kontex-neuro/pyxdaq.git
cd pyxdaq
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install .
```

### Windows
* Open PowerShell (By shift+right click) at where you want to install the project and enter the following commands.
```powershell
git clone https://github.com/kontex-neuro/pyxdaq.git
cd pyxdaq
python --version # check that Python 3.9 or higher is installed
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install .
```

## Running the self diagnosis script
```shell
python scripts/self_diagnosis.py
```
