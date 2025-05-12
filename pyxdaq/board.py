import json
import logging
from dataclasses import dataclass
from typing import Callable, List

from pylibxdaq import device, pyxdaq_device

from .constants import EndPoints

logger = logging.getLogger(__name__)


class BoardInfo(device.DeviceInfo):

    def create(self):
        manager = pyxdaq_device.get_device_manager(str(self.manager_path))
        raw = manager.create_device(json.dumps(self.options))
        status = json.loads(raw.get_status())
        info = json.loads(raw.get_info())
        return Board(self, raw, status, info, rhs=status['Mode'] == 'rhs')


@dataclass
class Board(device.Device):
    rhs: bool

    @staticmethod
    def list_devices() -> List[BoardInfo]:
        return [BoardInfo(**d.__dict__) for d in device.list_devices()]

    def GetWireOutValue(self, addr: EndPoints, update: bool = True) -> int:
        if update:
            return self.raw.get_register_sync(addr.value)
        else:
            return self.raw.get_register(addr.value)

    def SetWireInValue(
        self, addr: EndPoints, value: int, mask: int = 0xFFFFFFFF, update: bool = True
    ):
        if update:
            self.raw.set_register_sync(addr.value, value, mask)
        else:
            self.raw.set_register(addr.value, value, mask)

    def ActivateTriggerIn(self, addr: EndPoints, value: int):
        self.raw.trigger(addr.value, value)

    def WriteToBlockPipeIn(self, epAddr: EndPoints, data: bytearray):
        return self.raw.write(epAddr.value, data)

    def ReadFromBlockPipeOut(self, epAddr: EndPoints, data: bytearray):
        return self.raw.read(epAddr.value, data)

    def start_receiving_aligned_buffer(
        self,
        epAddr: EndPoints,
        alignment: int,
        callback: Callable[[pyxdaq_device.ManagedBuffer], None],
        chunk_size: int = 0
    ):
        return self.raw.start_aligned_read_stream(
            epAddr.value, alignment, callback, chunk_size=chunk_size
        )

    def SendTrig(
        self, trig: EndPoints, bit: int, epAddr: EndPoints, value: int, mask: int = 0xFFFFFFFF
    ):
        self.raw.set_register_sync(epAddr.value, value, mask)
        self.raw.trigger(trig.value, bit)
