import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

from pylibxdaq import pyxdaq_device
from pylibxdaq.managers import manager_paths

from .constants import EndPoints

logger = logging.getLogger(__name__)


@dataclass
class DeviceSetup:
    manager_path: Path
    manager_info: dict
    options: dict

    def with_mode(self, mode: str):
        info = deepcopy(self)
        info.options['mode'] = mode
        return info


class Board:

    @classmethod
    def list_devices(cls) -> List[DeviceSetup]:
        devices = []
        for manager_path in manager_paths:
            manager = pyxdaq_device.get_device_manager(str(manager_path))
            info = manager.info()
            for device_options in json.loads(manager.list_devices()):
                devices.append(DeviceSetup(manager_path, info, device_options))
        return sorted(devices, key=lambda x: str(x.options))

    def __init__(self, device_info: DeviceSetup):
        manager: pyxdaq_device.DeviceManager = pyxdaq_device.get_device_manager(
            str(device_info.manager_path)
        )
        self.dev = manager.create_device(json.dumps(device_info.options))
        status = json.loads(self.dev.get_status())
        if status['Mode'] == 'rhd':
            self.rhs = False
        elif status['Mode'] == 'rhs':
            self.rhs = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.dev

    def __getattr__(self, name: str):
        return getattr(self.dev, name)

    def GetWireOutValue(self, addr: EndPoints, update: bool = True) -> int:
        if update:
            return self.dev.get_register_sync(addr.value)
        else:
            return self.dev.get_register(addr.value)

    def SetWireInValue(
        self, addr: EndPoints, value: int, mask: int = 0xFFFFFFFF, update: bool = True
    ):
        if update:
            self.dev.set_register_sync(addr.value, value, mask)
        else:
            self.dev.set_register(addr.value, value, mask)

    def ActivateTriggerIn(self, addr: EndPoints, value: int):
        self.dev.trigger(addr.value, value)

    def WriteToBlockPipeIn(self, epAddr: EndPoints, data: bytearray):
        return self.dev.write(epAddr.value, data)

    def ReadFromBlockPipeOut(self, epAddr: EndPoints, data: bytearray):
        return self.dev.read(epAddr.value, data)

    def start_receiving_aligned_buffer(
        self,
        epAddr: EndPoints,
        alignment: int,
        callback: Callable[[pyxdaq_device.ManagedBuffer], None],
        chunk_size: int = 0
    ):
        return self.dev.start_aligned_read_stream(
            epAddr.value, alignment, callback, chunk_size=chunk_size
        )

    def SendTrig(
        self, trig: EndPoints, bit: int, epAddr: EndPoints, value: int, mask: int = 0xFFFFFFFF
    ):
        self.dev.set_register_sync(epAddr.value, value, mask)
        self.dev.trigger(trig.value, bit)
