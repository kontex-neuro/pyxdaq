import platform
import os
from pathlib import Path

if platform.system() == 'Darwin':
    cwd = Path.cwd()
    try:
        os.chdir(os.path.dirname(__file__))
        from . import ok
    finally:
        os.chdir(cwd)
else:
    from . import ok