from importlib.resources import files
from pathlib import Path


class rhd:
    bitfile_path = Path(files('pyxdaq.resources').joinpath('bitfiles').joinpath('xr7310a75.bit'))
    isa_path = Path(files('pyxdaq.resources').joinpath('config').joinpath('isa_rhd.json'))
    reg_path = Path(files('pyxdaq.resources').joinpath('config').joinpath('reg_rhd.json'))


class rhs:
    bitfile_path = Path(files('pyxdaq.resources').joinpath('bitfiles').joinpath('xsr7310a75.bit'))
    isa_path = Path(files('pyxdaq.resources').joinpath('config').joinpath('isa_rhs.json'))
    reg_path = Path(files('pyxdaq.resources').joinpath('config').joinpath('reg_rhs.json'))
