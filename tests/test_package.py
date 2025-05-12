import json


def test_package_import():
    import pyxdaq
    import pyxdaq.xdaq


def test_find_package_resources():
    from pyxdaq import resources
    for attr in ['isa_path', 'reg_path']:
        assert getattr(resources.rhd, attr).exists()
        assert getattr(resources.rhs, attr).exists()


def test_device_managers_loadable():
    from pylibxdaq import pyxdaq_device
    from pylibxdaq.managers import DeviceManagerPaths

    for path in DeviceManagerPaths:
        manager = pyxdaq_device.get_device_manager(str(path))
        info = manager.info()
        parsed = json.loads(info)
        assert "name" in parsed, f"Failed to load manager {path}"


def test_list_devices():
    from pyxdaq.board import Board
    devices = Board.list_devices()
    assert isinstance(devices, list)
