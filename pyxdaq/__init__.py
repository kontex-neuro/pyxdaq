from pathlib import Path

try:
    from .okFrontPanel import ok
except ImportError as e:
    class _OkMock:
        is_mock = True

        def __getattr__(self, name):
            raise RuntimeError(
                'Opal Kelly FrontPanel API is not installed. Please follow the instructions at README.md to install it.'
            )

    ok = _OkMock()