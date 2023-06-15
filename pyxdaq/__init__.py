import os
import platform
from pathlib import Path

try:
    if platform.system() == 'Darwin':

        def _macos_import_hack():
            cwd = os.getcwd()
            os.chdir(Path(__file__).parent.joinpath('okFrontPanel').resolve())
            from .okFrontPanel import ok
            os.chdir(cwd)
            return ok

        ok = _macos_import_hack()
    else:
        from .okFrontPanel import ok
except ImportError:
    class _OkMock:
        is_mock = True
        def __getattr__(self, name):
            raise RuntimeError('Opal Kelly FrontPanel API is not installed. Please follow the instructions at README.md to install it.')
    ok = _OkMock()
