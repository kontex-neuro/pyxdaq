from importlib.resources import files
from pathlib import Path


class rhd:
    isa_path = Path(files('pyxdaq.resources').joinpath('config').joinpath('isa_rhd.json'))
    reg_path = Path(files('pyxdaq.resources').joinpath('config').joinpath('reg_rhd.json'))


class rhs:
    isa_path = Path(files('pyxdaq.resources').joinpath('config').joinpath('isa_rhs.json'))
    reg_path = Path(files('pyxdaq.resources').joinpath('config').joinpath('reg_rhs.json'))
